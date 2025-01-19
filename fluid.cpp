#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_timer.h>

#define LINE_WIDTH 5

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
#ifdef __AVX__
		output[x * H + y] = Vector2(
			input[(x + 1) * H + y] - input[(x - 1) * H + y],
			input[x * H + y + 1] - input[x * H + y - 1]
		);
#else
		output[x * H + y] = Vector2(
			input[(x + 1) * H + y] - input[(x - 1) * H + y],
			input[x * H + y + 1] - input[x * H + y - 1]
		)*0.5;
#endif
	}
#ifdef __AVX__
	float *outf = reinterpret_cast<float*>(output);
	for (uint i = 0;i < 2*W*H/8;++i) {
		__m256 a = _mm256_load_ps(outf + i*8);
		__m256 mul = _mm256_set1_ps(0.5);
		__m256 result = _mm256_mul_ps(a,mul);
		_mm256_store_ps(outf + i*8, result);
	}
#endif
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

struct MouseState {
	Vector2 *down;
	Vector2 pos;
};

class PhyBox {
	Vector2 *v;		  // velocity
	Vector2 *tmpMem;  // used for calculations
	Vector2 *tmpMem2; // used for calculations
	alignas(32) float *p;		  // pressure
	float viscosity = 0.01;

	uint W, H;

  public:
	MouseState mouse;
	PhyBox(uint width, uint height) : W(width), H(height) {
		assert((W * H) % 8 == 0);
		v = new (std::align_val_t(32)) Vector2[W * H];
		tmpMem = new (std::align_val_t(32)) Vector2[W * H];
		tmpMem2 = new (std::align_val_t(32)) Vector2[W * H];
		p = new (std::align_val_t(32)) float[W * H];
		mouse = MouseState {
			down: nullptr,
			pos: Vector2(0,0),
		};
	}

	~PhyBox() {
		delete[] v;
		delete[] p;
		delete[] tmpMem;
	}

	void updateBuffer(SDL_Renderer *renderer, uint32_t *buffer) {
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
	}

	void impulse() {
		Vector2 diff = mouse.pos - *mouse.down;
		for (float i = 0;i < 1;i += 0.1) {
			Vector2 pt = *mouse.down + i*diff;
			for (int a = -LINE_WIDTH;a < LINE_WIDTH;++a)
			for (int b = -LINE_WIDTH;b < LINE_WIDTH;++b) {
				Vector2 ept = pt + Vector2(a,b);
				ept.x = min(max(0.0f,ept.x), (float)W-1);
				ept.y = min(max(0.0f,ept.y), (float)H-1);
				v[(uint32_t)ept.x*H + (uint32_t)ept.y] += diff;
			}
		}
	}

	void down(uint x, uint y) {
		mouse.down = new Vector2(x,y);
		mouse.pos.x = x;
		mouse.pos.y = y;
	}

	void up(uint x, uint y) {
		if (mouse.down == nullptr)
			return;
		impulse();
		delete mouse.down;
		mouse.down = nullptr;
		mouse.pos.x = x;
		mouse.pos.y = y;
	}

	void handle_motion(uint x, uint y) {
		if (mouse.down == nullptr)
			return;
		mouse.pos.x = x;
		mouse.pos.y = y;
	}

	void setBlocks(uint xs, uint ys, uint xe, uint ye) {
		for (uint i = xs; i < xe; ++i)
		for (uint j = ys; j < ye; ++j)
			v[i*H + j] = Vector2(0,0);
	}

	void forward(float dt = 1) {
		//setBlocks(200, 200, 400, 250);

		advect(tmpMem, dt);
		memcpy(v, tmpMem, W * H * sizeof(Vector2));

#ifdef DIFFUSE
		diffuse(tmpMem, dt);
#endif
		updatePressure(reinterpret_cast<float *>(tmpMem));

		gradient(tmpMem, p, W, H);
		subtract(v, tmpMem, W, H);
	}

	uint width() { return W; }
	uint height() { return H; }

  private:
#ifdef __AVX__
	void jacobi(float *x, float *b, float *mem, float alpha, float invbeta, uint iters = 50) {
		for (uint t = 0; t < iters; ++t) {
			for (uint i = 1; i < W - 1; ++i) {
				for (uint j = 1; j < H - 1; j+=1) {
					/*
					__m256 left  = _mm256_loadu_ps(&x[i * H + j - 1]);
					__m256 right = _mm256_loadu_ps(&x[i * H + j + 1]);
					__m256 up    = _mm256_loadu_ps(&x[(i - 1) * H + j]);
					__m256 down  = _mm256_loadu_ps(&x[(i + 1) * H + j]);
					__m256 center = _mm256_loadu_ps(&b[i * H + j]);
					__m256 sum = _mm256_add_ps(_mm256_add_ps(left, right), _mm256_add_ps(up, down));
					__m256 alpha_b = _mm256_mul_ps(_mm256_set1_ps(alpha), center);
					__m256 result = _mm256_add_ps(sum, alpha_b);
					result = _mm256_mul_ps(result, mbeta);
					_mm256_storeu_ps(&mem[i * H + j], result);
					*/

					mem[i * H + j] = (x[(i - 1) * H + j] + x[(i + 1) * H + j] + x[i * H + j - 1] + x[i * H + j + 1] + alpha * b[i * H + j]);
				}
			}
			for (uint i = 0;i < W*H/8;i++) {
				__m256 vals = _mm256_load_ps(mem + i*8);
				__m256 mbeta = _mm256_set1_ps(invbeta);
				__m256 res = _mm256_mul_ps(vals,mbeta);
				_mm256_store_ps(mem + i*8, res);
			}
			std::swap(x,mem);
		}
	}

	void jacobi(Vector2 *x, Vector2 *b, Vector2 *mem, float alpha, float invbeta, uint iters = 50) {
		for (uint t = 0; t < iters; ++t) {
			for (uint i = 1; i < W - 1; ++i) {
				for (uint j = 1; j < H - 1; ++j) {
					mem[i * H + j] = (x[(i - 1) * H + j] + x[(i + 1) * H + j] + x[i * H + j - 1] + x[i * H + j + 1] + alpha * b[i * H + j]);
				}
			}
			float *fmem = reinterpret_cast<float*>(mem);
			for (uint i = 0;i < 2*W*H/8;i++) {
				__m256 vals = _mm256_load_ps(fmem + i*8);
				__m256 mbeta = _mm256_set1_ps(invbeta);
				__m256 res = _mm256_mul_ps(vals,mbeta);
				_mm256_store_ps(fmem + i*8, res);
			}
			std::swap(x,mem);
		}
	}
#else
	template <typename T>
	void jacobi(T *x, T *b, T *mem, float alpha, float invbeta, uint iters = 50) {
		for (uint t = 0; t < iters; ++t) {
			for (uint i = 1; i < W - 1; ++i) {
				for (uint j = 1; j < H - 1; ++j) {
					mem[i * H + j] = (x[(i - 1) * H + j] + x[(i + 1) * H + j] + x[i * H + j - 1] + x[i * H + j + 1] + alpha * b[i * H + j]) * invbeta;
				}
			}
			std::swap(x,mem);
		}
	}
#endif

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
		jacobi(v, vecfield, tmpMem2, alpha, 1.0 / (4.0 + alpha), 30);
	}

	void updatePressure(float *mem) {
		divergence(mem);
		jacobi(p, mem, reinterpret_cast<float *>(tmpMem2), -1, 0.25, 30);
	}
};

struct Params {
	float viscosity;
	float shc;
};

void handle_click(SDL_MouseButtonEvent &e, PhyBox &pb, bool down) {
	if (e.button != 1)
		return;
	if (down)
		pb.down(e.y, e.x);
	else
		pb.up(e.y, e.x);
}

void handle_motion(SDL_MouseMotionEvent &e, PhyBox &pb) {
	pb.handle_motion(e.y, e.x);
}

int main() {
	Params params;
	params.viscosity = 0.005;
	params.shc = 1;
	PhyBox pb(600, 600);
	//PhyBox pb(1200, 1200);

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
				case SDL_MOUSEMOTION:
					handle_motion(event.motion, pb);
					break;
				case SDL_MOUSEBUTTONDOWN:
					handle_click(event.button, pb, true);
					break;
				case SDL_MOUSEBUTTONUP:
					handle_click(event.button, pb, false);
					break;
			}
		}

		pb.forward(0.3);
		pb.updateBuffer(renderer, buffer);

		SDL_UpdateTexture(texture, nullptr, buffer,
						  pb.width() * sizeof(uint32_t));
		SDL_RenderClear(renderer);
		SDL_RenderCopy(renderer, texture, nullptr, nullptr);
		if (pb.mouse.down != nullptr) {
			SDL_SetRenderDrawColor(renderer, 0xff, 0xff, 0xff, 0xff); // Example: Red line
			SDL_RenderDrawLine(renderer, pb.mouse.down->y, pb.mouse.down->x, pb.mouse.pos.y, pb.mouse.pos.x);
		}
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
