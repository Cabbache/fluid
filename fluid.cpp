#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_timer.h>

#include <cmath>
#include <limits>
#ifdef __AVX__
#include <immintrin.h>
#else
#warning avx is not enabled
#endif
#include <algorithm>
#include <iostream>

#include <cassert>

#include <chrono>

using namespace std;

typedef unsigned int uint;

struct alignas(8) Vector2 {
	float x, y;
	Vector2(float _x = 0, float _y = 0) : x(_x), y(_y) {}
	Vector2 operator+(const Vector2 &other) const {
		return Vector2(x + other.x, y + other.y);
	}
	Vector2 operator-(const Vector2 &other) const {
		return Vector2(x - other.x, y - other.y);
	}
	Vector2 operator+=(const Vector2 &other) {
		x += other.x;
		y += other.y;
		return *this;
	}
	Vector2 operator-=(const Vector2 &other) {
		x -= other.x;
		y -= other.y;
		return *this;
	}
	Vector2 operator*=(float c) {
		x *= c;
		y *= c;
		return *this;
	}
	Vector2 operator/=(float c) {
		x /= c;
		y /= c;
		return *this;
	}
	float mag2() { return x * x + y * y; }
	friend ostream &operator<<(ostream &os, const Vector2 &vec) {
		os << "(" << vec.x << ", " << vec.y << ")";
		return os;
	}
};
inline Vector2 operator-(const Vector2 &vec) { return Vector2(-vec.x, -vec.y); }
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
			// output[x*H + y] = Vector2(input[(x+1)*H + y] - input[x*H + y],
			// input[x*H + y+1] - input[x*H + y]);
		}
}

void subtract(Vector2 *dst_vec, Vector2 *src_vec, uint W, uint H) {
#ifdef __AVX__
	float *src = reinterpret_cast<float *>(src_vec);
	float *dst = reinterpret_cast<float *>(dst_vec);
	uint iters = 2 * W * H / 8; //*2 because vec has two floats
	for (uint i = 0; i < iters; ++i) {
		uint s = i * 8;
		__m256 a = _mm256_load_ps(dst + s);
		__m256 b = _mm256_load_ps(src + s);
		a = _mm256_sub_ps(a, b);
		_mm256_store_ps(dst + s, a);
	}
#else
	for (uint i = 0; i < W * H; ++i)
		dst_vec[i] -= src_vec[i];
#endif
}

class PhyBox {
	Vector2 *v;		  // velocity
	Vector2 *tmpMem;  // used for calculations
	Vector2 *tmpMem2; // used for calculations
	float *p;		  // pressure
	float viscosity = 0.01;

	uint W, H;

  public:
	PhyBox(uint width, uint height) : W(width), H(height) {
		assert((W * H) % 8 == 0);
		v = new (std::align_val_t(32)) Vector2[W * H];
		tmpMem = new (std::align_val_t(32)) Vector2[W * H];
		tmpMem2 = new Vector2[W * H];
		p = new float[W * H];
	}

	~PhyBox() {
		delete[] v;
		delete[] p;
		delete[] tmpMem;
	}

	void updateBuffer(uint32_t *buffer) {
		float max_x, min_x, max_y, min_y;

#ifdef SCALE
		max_x = max_y = -std::numeric_limits<float>::infinity();
		min_x = min_y = std::numeric_limits<float>::infinity();
		for (uint i = 0; i < W * H; ++i) {
			max_x = max(max_x, v[i].x);
			min_x = min(min_x, v[i].x);
			max_y = max(max_y, v[i].y);
			min_y = min(min_y, v[i].y);
		}
#else
		max_x = max_y = 60;
		min_x = min_y = -20;
#endif

		float sx = 255 / (max_x - min_x);
		float sy = 255 / (max_y - min_y);

		for (uint i = 0; i < W * H; ++i) {
			uint8_t r = static_cast<uint8_t>(sx * (v[i].x - min_x));
			uint8_t g = static_cast<uint8_t>(sy * (v[i].y - min_y));
			buffer[i] = (r << 16) | (g << 8) | 0x00;
		}

		/*
		for (uint y = 0; y < H; ++y)
		for (uint x = 0; x < W; ++x) {
			uint8_t r = static_cast<uint8_t>(sx*(v[x*H+y].x - min_x));
			uint8_t g = static_cast<uint8_t>(sy*(v[x*H+y].y - min_y));
			//uint8_t b = static_cast<uint8_t>(mag2i % 256);
			buffer[y*W + x] = (r << 16) | (g << 8) | 0x00;
		}
		*/
	}

	void impulse(uint x, uint y, uint fx, uint fy) {
		for (uint i = 0; i < 100; ++i)
			for (uint j = 0; j < 100; ++j)
				v[(min(x + i, W - 1)) * H + min(y + j, H - 1)] +=
					Vector2(fx, fy);
	}

	void setBlocks(uint xs, uint ys, uint xe, uint ye) {
		for (uint i = xs; i < xe; ++i)
		for (uint j = ys; j < ye; ++j)
			v[i*H + j] = Vector2(0,0);
	}

	void forward(float dt = 1) {
		setBlocks(200, 200, 400, 250);

		advect(tmpMem, dt);
		memcpy(v, tmpMem, W * H * sizeof(Vector2));

#ifdef DIFFUSE
		diffuse(tmpMem, dt);
#endif
		updatePressure((float *)tmpMem);

		gradient(tmpMem, p, W, H);
		subtract(v, tmpMem, W, H);
	}

	uint width() { return W; }
	uint height() { return H; }

  private:
	template <typename T>
	void jacobi(T *x, T *b, T *mem, float alpha, float beta, uint iters = 50) {
		for (uint t = 0; t < iters; ++t) {
			for (uint i = 1; i < W - 1; ++i)
				for (uint j = 1; j < H - 1; ++j) {
					mem[i * H + j] = (x[(i - 1) * H + j] + x[(i + 1) * H + j] +
									  x[i * H + j - 1] + x[i * H + j + 1] +
									  alpha * b[i * H + j]) /
									 beta;
				}
			memcpy(x, mem, sizeof(T) * W * H);
		}
	}

	Vector2 lerp_index(const Vector2 &index) {
		int xl = (int)floor(index.x);
		int yl = (int)floor(index.y);
#ifdef LERP
		int xh = (int)ceil(index.x);
		int yh = (int)ceil(index.y);

		// TODO
		xl = min(max(xl, 0), (int)W - 1);
		xh = min(max(xh, 0), (int)W - 1);
		yl = min(max(yl, 0), (int)H - 1);
		yh = min(max(yh, 0), (int)H - 1);

		//assert(xl >= 0 && xl < W);
		//assert(xh >= 0 && xh < W);
		//assert(yl >= 0 && yl < H);
		//assert(yh >= 0 && yh < H);

		float dx = index.x - xl;
		float dy = index.y - yl;
		Vector2 myl = dx * v[xh * H + yl] + (1 - dx) * v[xl * H + yl];
		Vector2 myh = dx * v[xh * H + yh] + (1 - dx) * v[xl * H + yh];
		return dy * myh + (1 - dy) * myl;
#else
		return v[xl * H + yl];
#endif
	}

	void advect(Vector2 *result, float dt = 1) {
		for (uint x = 3; x < W - 3; ++x)
			for (uint y = 3; y < H - 3; ++y) {
				Vector2 vel = v[x * H + y];
				result[x * H + y] = lerp_index(Vector2(x, y) - vel * dt);
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

	void diffuse(Vector2 *vecfield, float dt = 1) {
		float alpha = 1 / (viscosity * dt);
		memcpy(vecfield, v, W * H * sizeof(Vector2));
		jacobi(v, vecfield, tmpMem2, alpha, 4 + alpha);
	}

	void updatePressure(float *mem) {
		divergence(mem);
		jacobi(p, mem, (float *)tmpMem2, -1, 4);
	}
};

struct Params {
	float viscosity;
	float shc;
};

void handle_click(SDL_MouseButtonEvent &e, PhyBox &pb) {
	switch (e.button) {
		case 1: // left click
			pb.impulse(e.y, e.x, 30, 0);
			break;
		case 3: // right click
			pb.impulse(e.y, e.x, 0, 30);
			break;
	}
}

int main() {
	Params params;
	params.viscosity = 0.005;
	params.shc = 1;
	PhyBox pb(600, 600);

	if (SDL_Init(SDL_INIT_EVERYTHING) != 0)
		printf("error initializing SDL: %s\n", SDL_GetError());
	SDL_Window *window =
		SDL_CreateWindow("FLUID", SDL_WINDOWPOS_CENTERED,
						 SDL_WINDOWPOS_CENTERED, pb.width(), pb.height(), 0);

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
	memset(buffer, 0, pb.width() * pb.height());

	bool running = true;
	SDL_Event event;
	uint thresh = 100;

	std::chrono::steady_clock::time_point tick =
		std::chrono::steady_clock::now();
	uint ctr = 0;
	while (running) {
		while (SDL_PollEvent(&event)) {
			switch (event.type) {
				case SDL_QUIT:
					running = false;
					break;
				case SDL_MOUSEBUTTONDOWN:
					handle_click(event.button, pb);
					break;
			}
		}

		pb.forward(0.3);
		pb.updateBuffer(buffer);

		SDL_UpdateTexture(texture, nullptr, buffer,
						  pb.width() * sizeof(uint32_t));
		SDL_RenderClear(renderer);
		SDL_RenderCopy(renderer, texture, nullptr, nullptr);
		SDL_RenderPresent(renderer);

		if (++ctr % thresh == 0) {
			std::chrono::steady_clock::time_point now =
				std::chrono::steady_clock::now();
			uint us = std::chrono::duration_cast<std::chrono::microseconds>(
						  now - tick)
						  .count();
			cout << "fps: " << thresh * 1000000.0 / us << endl;
			tick = std::chrono::steady_clock::now();
		}
	}

	// Clean up
	delete[] buffer;
	SDL_DestroyTexture(texture);
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();
	return 0;
}
