#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_timer.h>

#include <cmath>
#include <immintrin.h>
#include <iostream>

#include <cassert>

using namespace std;

typedef unsigned int uint;

void updateBuffer(uint32_t *buffer, int width, int height, int tick) {
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			// Create a simple animation pattern by manipulating pixel values
			uint8_t r = static_cast<uint8_t>((x + tick) % 256);
			uint8_t g = static_cast<uint8_t>((y + tick) % 256);
			uint8_t b = static_cast<uint8_t>((x + y + tick) % 256);
			buffer[y * width + x] =
				(r << 16) | (g << 8) | b; // Pack RGB into a 32-bit integer
		}
	}
}

struct Vector2 {
	float x, y;
	Vector2(float _x = 0, float _y = 0) : x(_x), y(_y) {}
	Vector2 operator+(const Vector2 &other) const {
		return Vector2(x + other.x, y + other.y);
	}
	Vector2 operator-(const Vector2 &other) const {
		return Vector2(x - other.x, y - other.y);
	}
	friend ostream &operator<<(ostream &os, const Vector2 &vec) {
		os << "(" << vec.x << ", " << vec.y << ")";
		return os;
	}
};
inline Vector2 operator*(float s, const Vector2 &vec) {
	return Vector2(vec.x * s, vec.y * s);
}
inline Vector2 operator*(const Vector2 &vec, float s) {
	return Vector2(vec.x * s, vec.y * s);
}
inline Vector2 operator/(float s, const Vector2 &vec) {
	return Vector2(vec.x / s, vec.y / s);
}
inline Vector2 operator/(const Vector2 &vec, float s) {
	return Vector2(vec.x / s, vec.y / s);
}

void gradient(Vector2 *output, float *input, uint W, uint H) {
	for (uint x = 1; x < W - 1; ++x)
		for (uint y = 1; y < H - 1; ++y) {
			output[x * H + y] =
				Vector2(input[(x + 1) * H + y] - input[(x - 1) * H + y],
						input[x * H + y + 1] - input[x * H + y - 1]) /
				2;
		}
}

void subtract(Vector2 *dst, Vector2 *src, uint W, uint H) {
	for (uint i = 0; i < W * H; ++i)
		dst[i] = dst[i] - src[i];
}

class PhyBox {
	Vector2 *v; // velocity
	float *p;	// pressure
	float viscosity = 1;

	uint W, H;

  public:
	PhyBox(uint width, uint height) : W(width), H(height) {
		v = new Vector2[W * H];
		p = new float[W * H];
	}

	void forward(float dt = 1) {
		Vector2 *adv_res = new Vector2[W * H];
		advect(adv_res, dt);
		memcpy(v, adv_res, W * H * sizeof(Vector2));
		delete[] adv_res;

		diffuse(dt);
		updatePressure();

		Vector2 *pressure_gradient = new Vector2[W * H];
		gradient(pressure_gradient, p, W, H);
		subtract(v, pressure_gradient, W, H);
		delete[] pressure_gradient;
	}

	uint width() { return W; }
	uint height() { return H; }

  private:
	template <typename T>
	void jacobi(T *x, T *b, float alpha, float beta, uint iters = 20) {
		// TODO pass memory in arg instead of allocating here
		T *result = new T[W * H];
		for (uint t = 0; t < iters; ++t) {
			for (uint i = 1; i < W - 1; ++i)
				for (uint j = 1; j < H - 1; ++j) {
					result[i * H + j] =
						(x[(i - 1) * H + j] + x[(i + 1) * H + j] +
						 x[i * H + j - 1] + x[i * H + j + 1] +
						 alpha * b[i * H + j]) /
						beta;
				}
			memcpy(x, result, sizeof(T) * W * H);
		}
		delete[] result;
	}

	Vector2 lerp_index(const Vector2 &index) {
		uint xl = (uint)floor(index.x);
		uint xh = (uint)ceil(index.x);
		uint yl = (uint)floor(index.y);
		uint yh = (uint)ceil(index.y);

		assert(xl >= 0 && xl < W);
		assert(xh >= 0 && xh < W);
		assert(yl >= 0 && yl < H);
		assert(yh >= 0 && yh < H);

		float dx = index.x - xl;
		float dy = index.y - yl;

		Vector2 myl = dx * v[xh * H + yl] + (1 - dx) * v[xl * H + yl];
		Vector2 myh = dx * v[xh * H + yh] + (1 - dx) * v[xl * H + yh];
		return dy * myh + (1 - dy) * myl;
	}

	void advect(Vector2 *result, float dt = 1) {
		for (uint x = 0; x < W; ++x)
			for (uint y = 0; y < H; ++y) {
				Vector2 vel = v[x * H + y];
				result[x * H + y] =
					lerp_index(Vector2(x - vel.x, y - vel.y)) * dt;
			}
	}

	void divergence(float *result) {
		for (uint x = 1; x < W - 1; ++x)
			for (uint y = 1; y < H - 1; ++y) {
				result[x * H + y] =
					(v[(x + 1) * H + y].x - v[(x - 1) * H + y].x +
					 v[x * H + y + 1].y - v[x * H + y - 1].y) /
					2;
			}
	}

	void diffuse(float dt = 1) {
		float alpha = 1 / (viscosity * dt);
		Vector2 *b = new Vector2[W * H];
		memcpy(b, v, W * H * sizeof(Vector2));
		jacobi(v, b, alpha, 4 + alpha);
		delete[] b;
	}

	void updatePressure() {
		float *dvg = new float[W * H];
		divergence(dvg);
		jacobi(p, dvg, -1, 4);
	}
};

struct Params {
	float viscosity;
	float shc;
};

int main() {
	Params params;
	params.viscosity = 0.005;
	params.shc = 1;

	PhyBox pb(800, 600);
	pb.forward();
	pb.forward();
	pb.forward();

	if (SDL_Init(SDL_INIT_EVERYTHING) != 0)
		printf("error initializing SDL: %s\n", SDL_GetError());
	SDL_Window *window =
		SDL_CreateWindow("GAME", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
						 pb.width(), pb.height(), 0);

	SDL_Renderer *renderer =
		SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
	if (!renderer) {
		std::cerr << "SDL_CreateRenderer Error: " << SDL_GetError()
				  << std::endl;
		SDL_DestroyWindow(window);
		SDL_Quit();
		return 1;
	}

	// Create a texture to render the buffer to
	SDL_Texture *texture =
		SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888,
						  SDL_TEXTUREACCESS_STREAMING, pb.width(), pb.height());
	if (!texture) {
		std::cerr << "SDL_CreateTexture Error: " << SDL_GetError() << std::endl;
		SDL_DestroyRenderer(renderer);
		SDL_DestroyWindow(window);
		SDL_Quit();
		return 1;
	}

	// Create a pixel buffer
	uint32_t *buffer = new uint32_t[pb.width() * pb.height()];

	for (int i = 20; i < 40; ++i)
		for (int j = 200; j < 220; ++j)
			buffer[i + j * pb.height()] = (255 | (255 << 8));

	bool running = true;
	SDL_Event event;
	int tick = 0;

	while (running) {
		while (SDL_PollEvent(&event)) {
			if (event.type == SDL_QUIT) {
				running = false;
			}
		}

		for (int i = 0; i < pb.width(); ++i)
			for (int j = 0; j < pb.height(); ++j)
				buffer[i + j * pb.height()] +=
					buffer[i + 1 + (j + 1) * pb.height()];
		SDL_UpdateTexture(texture, nullptr, buffer,
						  pb.width() * sizeof(uint32_t));
		SDL_RenderClear(renderer);
		SDL_RenderCopy(renderer, texture, nullptr, nullptr);
		SDL_RenderPresent(renderer);
		SDL_Delay(16);
		tick++;
	}

	// Clean up
	delete[] buffer;
	SDL_DestroyTexture(texture);
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();
	return 0;
}