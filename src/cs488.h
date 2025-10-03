// =======================================
// CS488/688 base code
// (written by Toshiya Hachisuka)
// =======================================
#pragma once
#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX


// OpenGL
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>


// image loader and writer
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "stb_image_resize.h"


// linear algebra 
#include "linalg.h"
#include "ltc.h"
using namespace linalg::aliases;


// animated GIF writer
#include "gif.h"


// misc
#include <iostream>
#include <vector>
#include <cfloat>
#include <chrono>
#include <algorithm>
#include <random>


// main window
static GLFWwindow* globalGLFWindow;


// window size and resolution
// (do not make it too large - will be slow!)
constexpr int globalWidth = 512;
constexpr int globalHeight = 384;


// degree and radian
constexpr float PI = 3.14159265358979f;
constexpr float DegToRad = PI / 180.0f;
constexpr float RadToDeg = 180.0f / PI;


// for ray tracing
constexpr float Epsilon = 1.1e-5f;
bool enableEnvironmentMapping = false;


// amount the camera moves with a mouse and a keyboard
constexpr float ANGFACT = 0.2f;
constexpr float SCLFACT = 0.1f;


// fixed camera parameters
constexpr float globalAspectRatio = float(globalWidth / float(globalHeight));
constexpr float globalFOV = 45.0f; // vertical field of view
constexpr float globalDepthMin = Epsilon; // for rasterization
constexpr float globalDepthMax = 100.0f; // for rasterization
constexpr float globalFilmSize = 0.032f; //for ray tracing
const float globalDistanceToFilm = globalFilmSize / (2.0f * tan(globalFOV * DegToRad * 0.5f)); // for ray tracing


// particle system related
bool globalEnableParticles = false;
constexpr float deltaT = 0.002f;
constexpr float3 globalGravity = float3(0.0f, -9.8f, 0.0f);
constexpr int globalNumParticles = 15;


// dynamic camera parameters
float3 globalEye = float3(0.0f, 0.0f, 1.5f);
float3 globalLookat = float3(0.0f, 0.0f, 0.0f);
float3 globalUp = normalize(float3(0.0f, 1.0f, 0.0f));
float3 globalViewDir; // should always be normalize(globalLookat - globalEye)
float3 globalRight; // should always be normalize(cross(globalViewDir, globalUp));
bool globalShowRaytraceProgress = false; // for ray tracing


// matrix for scaling the ndc coords to the size of the screen for pixel coords
float4x4 ndcFlattenMatrix = {
	{ globalWidth / 2.0f, 0.f, 0.f, 0.f },
	{ 0.f, globalHeight / 2.0f, 0.f, 0.f },
	{ 0.f, 0.f, 1.0f, 0.f },
	{ (globalWidth - 1) / 2.0f, (globalHeight - 1) / 2.0f, 0.f, 1.0f }
};


// mouse event
static bool mouseLeftPressed;
static double m_mouseX = 0.0;
static double m_mouseY = 0.0;


// rendering algorithm
enum enumRenderType {
	RENDER_RASTERIZE,
	RENDER_RAYTRACE,
	RENDER_IMAGE,
};
enumRenderType globalRenderType = RENDER_IMAGE;
int globalFrameCount = 0;
static bool globalRecording = false;
static GifWriter globalGIFfile;
constexpr int globalGIFdelay = 1;

enum enumAreaLightMethod {
	AREA_LIGHT_LTC,
	AREA_LIGHT_MONTE_CARLO,
};
enumAreaLightMethod globalAreaLightMethod = AREA_LIGHT_LTC;

float globalRoughness = 0.0f;
bool globalUseMipMap = true;
bool globalUseFresnel = true;
bool antiAliasMinv = true;
bool globalParticleConstraint = true;
bool testCollisions = false;

// OpenGL related data (do not modify it if it is working)
static GLuint GLFrameBufferTexture;
static GLuint FSDraw;
static const std::string FSDrawSource = R"(
    #version 120

    uniform sampler2D input_tex;
    uniform vec4 BufInfo;

    void main()
    {
        gl_FragColor = texture2D(input_tex, gl_FragCoord.st * BufInfo.zw);
    }
)";
static const char* PFSDrawSource = FSDrawSource.c_str();



// fast random number generator based pcg32_fast
#include <stdint.h>
namespace PCG32 {
	static uint64_t mcg_state = 0xcafef00dd15ea5e5u;	// must be odd
	static uint64_t const multiplier = 6364136223846793005u;
	uint32_t pcg32_fast(void) {
		uint64_t x = mcg_state;
		const unsigned count = (unsigned)(x >> 61);
		mcg_state = x * multiplier;
		x ^= x >> 22;
		return (uint32_t)(x >> (22 + count));
	}
	float rand() {
		return float(double(pcg32_fast()) / 4294967296.0);
	}
}



// image with a depth buffer
// (depth buffer is not always needed, but hey, we have a few GB of memory, so it won't be an issue...)
class Image {
public:
	std::vector<float3> pixels;
	std::vector<float> depths;
	int width = 0, height = 0;

	static float toneMapping(const float r) {
		// you may want to implement better tone mapping
		return std::max(std::min(1.0f, r), 0.0f);
	}

	static float gammaCorrection(const float r, const float gamma = 1.0f) {
		// assumes r is within 0 to 1
		// gamma is typically 2.2, but the default is 1.0 to make it linear
		return pow(r, 1.0f / gamma);
	}

	void resize(const int newWdith, const int newHeight) {
		this->pixels.resize(newWdith * newHeight);
		this->depths.resize(newWdith * newHeight);
		this->width = newWdith;
		this->height = newHeight;
	}

	void clear() {
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				this->pixel(i, j) = float3(0.0f);
				this->depth(i, j) = FLT_MAX;
			}
		}
	}

	Image(int _width = 0, int _height = 0) {
		this->resize(_width, _height);
		this->clear();
	}

	bool valid(const int i, const int j) const {
		return (i >= 0) && (i < this->width) && (j >= 0) && (j < this->height);
	}

	float& depth(const int i, const int j) {
		return this->depths[i + j * width];
	}

	float3& pixel(const int i, const int j) {
		// optionally can check with "valid", but it will be slow
		return this->pixels[i + j * width];
	}

	void load(const char* fileName, int& err) {
		int comp, w, h;
		float* buf = stbi_loadf(fileName, &w, &h, &comp, 3);
		if (!buf) {
			std::cerr << "Unable to load: " << fileName << std::endl;
			err = 1;
			return;
		}

		this->resize(w, h);
		int k = 0;
		for (int j = height - 1; j >= 0; j--) {
			for (int i = 0; i < width; i++) {
				this->pixels[i + j * width] = float3(buf[k], buf[k + 1], buf[k + 2]);
				k += 3;
			}
		}
		delete[] buf;
		printf("Loaded \"%s\".\n", fileName);
		err = 0;
	}
	void save(const char* fileName) {
		unsigned char* buf = new unsigned char[width * height * 3];
		int k = 0;
		for (int j = height - 1; j >= 0; j--) {
			for (int i = 0; i < width; i++) {
				buf[k++] = (unsigned char)(255.0f * gammaCorrection(toneMapping(pixel(i, j).x)));
				buf[k++] = (unsigned char)(255.0f * gammaCorrection(toneMapping(pixel(i, j).y)));
				buf[k++] = (unsigned char)(255.0f * gammaCorrection(toneMapping(pixel(i, j).z)));
			}
		}
		stbi_write_png(fileName, width, height, 3, buf, width * 3);
		delete[] buf;
		printf("Saved \"%s\".\n", fileName);
	}
};

// main image buffer to be displayed
Image FrameBuffer(globalWidth, globalHeight);

// you may want to use the following later for progressive ray tracing
Image AccumulationBuffer(globalWidth, globalHeight);
unsigned int sampleCount = 0;


bool NO_COLLIDE_FLAG = false;
// keyboard events (you do not need to modify it unless you want to)
void keyFunc(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_PRESS || action == GLFW_REPEAT) {
		switch (key) {
			case GLFW_KEY_R: {
				if (globalRenderType == RENDER_RAYTRACE) {
					printf("(Switched to rasterization)\n");
					glfwSetWindowTitle(window, "Rasterization mode");
					globalRenderType = RENDER_RASTERIZE;
				} else if (globalRenderType == RENDER_RASTERIZE) {
					printf("(Switched to ray tracing)\n");
					AccumulationBuffer.clear();
					sampleCount = 0;
					glfwSetWindowTitle(window, "Ray tracing mode");
					globalRenderType = RENDER_RAYTRACE;
				}
			break;}

			case GLFW_KEY_0: {
				NO_COLLIDE_FLAG = !NO_COLLIDE_FLAG;
				break;
			}
			case GLFW_KEY_ESCAPE: {
				glfwSetWindowShouldClose(window, GL_TRUE);
			break;}

			case GLFW_KEY_I: {
				char fileName[1024];
				sprintf(fileName, "output%d.png", int(1000.0 * PCG32::rand()));
				FrameBuffer.save(fileName);
			break;}

			case GLFW_KEY_6: {
				if (globalRoughness < 1.0f) {
					globalRoughness += 0.01f;
					printf("Increasing roughness to %f\n", globalRoughness);
				}
				else {
					printf("Max roughness reached (1.0)\n");
				}
				break;
			}

			case GLFW_KEY_5: {
				if (globalRoughness > 0.0f) {
					globalRoughness -= 0.01f;
					printf("Decreasing roughness to %f\n", globalRoughness);
				}
				else {
					printf("Min roughness reached (0.0)\n");
				}
				break;
			}

			case GLFW_KEY_M: {
				if (!globalUseMipMap) {
					printf("Enabling mipmaps\n");
					globalUseMipMap = true;
				} else {
					printf("Disabling mipmaps\n");
					globalUseMipMap = false;
				}
				break;
			}

			case GLFW_KEY_N: {
				if (!globalUseFresnel) {
					printf("Enabling Fresnel glass reflection\n");
					globalUseFresnel = true;
				}
				else {
					printf("Disabling Fresnel glass reflection\n");
					globalUseFresnel = false;
				}
				break;
			}

			case GLFW_KEY_K: {
				if (!antiAliasMinv) {
					printf("Turning on LTC texture antialiasing\n");
					antiAliasMinv = true;
				}
				else {
					printf("Turning off LTC texture antialiasing\n");
					antiAliasMinv = false;
				}
				break;
			}

			case GLFW_KEY_C: {
				if (!globalParticleConstraint) {
					printf("Enabling particle constraint\n");
					globalParticleConstraint = true;
				}
				else {
					printf("Disabling particle constraint\n");
					globalParticleConstraint = false;
				}
				break;
			}


			case GLFW_KEY_F: {
				if (!globalRecording) {
					char fileName[1024];
					sprintf(fileName, "output%d.gif", int(1000.0 * PCG32::rand()));
					printf("Saving \"%s\"...\n", fileName);
					GifBegin(&globalGIFfile, fileName, globalWidth, globalHeight, globalGIFdelay);
					globalRecording = true;
					printf("(Recording started)\n");
				} else {
					GifEnd(&globalGIFfile);
					globalRecording = false;
					printf("(Recording done)\n");
				}
			break;}

			case GLFW_KEY_W: {
				globalEye += SCLFACT * globalViewDir;
				globalLookat += SCLFACT * globalViewDir;
			break;}

			case GLFW_KEY_S: {
				globalEye -= SCLFACT * globalViewDir;
				globalLookat -= SCLFACT * globalViewDir;
			break;}

			case GLFW_KEY_Q: {
				globalEye += SCLFACT * globalUp;
				globalLookat += SCLFACT * globalUp;
			break;}

			case GLFW_KEY_Z: {
				globalEye -= SCLFACT * globalUp;
				globalLookat -= SCLFACT * globalUp;
			break;}

			case GLFW_KEY_A: {
				globalEye -= SCLFACT * globalRight;
				globalLookat -= SCLFACT * globalRight;
			break;}

			case GLFW_KEY_D: {
				globalEye += SCLFACT * globalRight;
				globalLookat += SCLFACT * globalRight;
			break;}

			case GLFW_KEY_T: {
				if (globalAreaLightMethod == AREA_LIGHT_LTC) {
					printf("(Switched to Monte Carlo area light calculation)\n");
					globalAreaLightMethod = AREA_LIGHT_MONTE_CARLO;
				} else if (globalAreaLightMethod == AREA_LIGHT_MONTE_CARLO) {
					printf("(Switched to Linearly Transformed Cosine area light calculation)\n");
					globalAreaLightMethod = AREA_LIGHT_LTC;
				}
			}

			default: break;
		}
	}
}



// mouse button events (you do not need to modify it unless you want to)
void mouseButtonFunc(GLFWwindow* window, int button, int action, int mods) {
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		if (action == GLFW_PRESS) {
			mouseLeftPressed = true;
		} else if (action == GLFW_RELEASE) {
			mouseLeftPressed = false;
			if (globalRenderType == RENDER_RAYTRACE) {
				AccumulationBuffer.clear();
				sampleCount = 0;
			}
		}
	}
}



// mouse button events (you do not need to modify it unless you want to)
void cursorPosFunc(GLFWwindow* window, double mouse_x, double mouse_y) {
	if (mouseLeftPressed) {
		const float xfact = -ANGFACT * float(mouse_y - m_mouseY);
		const float yfact = -ANGFACT * float(mouse_x - m_mouseX);
		float3 v = globalViewDir;

		// local function in C++...
		struct {
			float3 operator()(float theta, const float3& v, const float3& w) {
				const float c = cosf(theta);
				const float s = sinf(theta);

				const float3 v0 = dot(v, w) * w;
				const float3 v1 = v - v0;
				const float3 v2 = cross(w, v1);

				return v0 + c * v1 + s * v2;
			}
		} rotateVector;

		v = rotateVector(xfact * DegToRad, v, globalRight);
		v = rotateVector(yfact * DegToRad, v, globalUp);
		globalViewDir = v;
		globalLookat = globalEye + globalViewDir;
		globalRight = cross(globalViewDir, globalUp);

		m_mouseX = mouse_x;
		m_mouseY = mouse_y;

		if (globalRenderType == RENDER_RAYTRACE) {
			AccumulationBuffer.clear();
			sampleCount = 0;
		}
	} else {
		m_mouseX = mouse_x;
		m_mouseY = mouse_y;
	}
}

const int numSpectrumSamples = 20;
const int minLambda = 380;
const int maxLambda = 780;
float3 XYZFunctionLookup[numSpectrumSamples];
const float3x3 XYZtoRGBMatrix = {
	{3.2404542f, -0.9692660f, 0.0556434f},
	{-1.5371385f, 1.8760108f, -0.2040259f},
	{-0.4985314f, 0.0415560f, 1.0572252f}
};

const float redSpectrumVals[numSpectrumSamples] = {0.089f, 0.089f, 0.0855f, 0.065f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.085f, 0.7f, 0.7f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
const float greenSpectrumVals[numSpectrumSamples] = {0.0f, 0.0f, 0.0f, 0.0f, 0.032f, 0.385f, 0.785f, 1.0f, 1.0f, 0.775f, 0.314f, 0.314f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
const float blueSpectrumVals[numSpectrumSamples] = {1.0f, 1.0f, 1.0f, 1.0f, 0.727f, 0.512f, 0.258f, 0.063f, 0.0f, 0.0f, 0.0f, 0.0f, 0.024f, 0.044f, 0.051f, 0.055f, 0.055f, 0.055f, 0.055, 0.055f};
const float cyanSpectrumVals[numSpectrumSamples] = {0.99f, 0.985f, 0.926f, 0.985f, 1.0f, 1.0f, 1.0f, 1.0f, 0.915f, 0.298f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
const float magentaSpectrumVals[numSpectrumSamples] = {1.0f, 1.0f, 1.0f, 1.0f, 0.973f, 0.61f, 0.195f, 0.0f, 0.0f, 0.244f, 0.46f, 0.679f, 0.98f, 1.0f, 1.0f, 1.0f, 0.985f, 0.978f, 0.973f, 0.975f};
const float yellowSpectrumVals[numSpectrumSamples] = {0.0f, 0.0f, 0.0f, 0.0f, 0.134f, 0.394f, 0.691f, 0.925f, 1.0f, 1.0f, 1.0f, 1.0f, 0.965f, 0.953f, 0.953f, 0.956f, 0.963f, 0.97f, 0.977f, 0.984f};
const float whiteSpectrumVals[numSpectrumSamples] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

float CIEGaussian(float x, float mu, float gamma1, float gamma2) {
	if (x < mu) {
		return std::exp(-std::pow(gamma1 * (x - mu), 2.0f) / 2);
	} else {
		return std::exp(-std::pow(gamma2 * (x - mu), 2.0f) / 2);
	}
}

void generateSpectrumXYZLookup() {
	int lambdaStep = (maxLambda - minLambda) / numSpectrumSamples;
	for (int i = 0; i < numSpectrumSamples; i++) {
		int wavelength = minLambda + lambdaStep * i;
		float x = 1.056f * CIEGaussian(wavelength, 599.8f, 0.0264f, 0.0323f) +
			0.362f * CIEGaussian(wavelength, 442.0f, 0.0624f, 0.0374f) -
				0.065f * CIEGaussian(wavelength, 501.1f, 0.0490f, 0.0382f);
		float y = 0.821f * CIEGaussian(wavelength, 568.8f, 0.0213f, 0.0247f) +
			0.286f * CIEGaussian(wavelength, 530.9f, 0.0613f, 0.0322f);
		float z = 1.217f * CIEGaussian(wavelength, 437.0f, 0.0845f, 0.0278f) +
			0.681f * CIEGaussian(wavelength, 459.0f, 0.0385f, 0.0725f);
		XYZFunctionLookup[i] = {x, y, z};
	}
}


class Spectrum {
public:

	Spectrum() {
		std::fill_n(c, numSpectrumSamples, 1.0f);
	}

	Spectrum(const float (*l) [numSpectrumSamples]) {
		for (int i = 0; i < numSpectrumSamples; i++) {
			c[i] = (*l)[i];
		}
	}

	Spectrum(const float f) {
		std::fill_n(c, numSpectrumSamples, f);
	}

	Spectrum(float* l) {
		for (int i = 0; i < numSpectrumSamples; i++) {
			c[i] = l[i];
		}
	}

	float* get_SPD(){return c;}

	Spectrum operator+(const Spectrum& s2) const {
		Spectrum ret{0.0f};
		for (int i = 0; i < numSpectrumSamples; i++) {
			ret.c[i] = c[i] + s2.c[i];
		}
		return ret;
	}

	Spectrum operator+=(const Spectrum& s2) {
		for (int i = 0; i < numSpectrumSamples; i++) {
			c[i] += s2.c[i];
		}
		return *this;
	}

	Spectrum operator*(const Spectrum& s2) const {
		Spectrum ret;
		for (int i = 0; i < numSpectrumSamples; i++) {
			ret.c[i] = c[i] * s2.c[i];
		}
		return ret;
	}

	Spectrum operator*(const float& f) const {
		Spectrum ret;
		for (int i = 0; i < numSpectrumSamples; i++) {
			ret.c[i] = c[i] * f;
		}
		return ret;
	}

	Spectrum operator*=(const float& f) {
		for (int i = 0; i < numSpectrumSamples; i++) {
			c[i] *= f;
		}
		return *this;
	}

	float3 convertToRGB() const {
		float3 xyz = {0.0f, 0.0f, 0.0f};
		for (int i = 0; i < numSpectrumSamples; i++) {
			xyz += c[i] * XYZFunctionLookup[i];
			std::cout << xyz[0] << " " << xyz[1] << " " << xyz[2] << std::endl;
		}

		float3 rgb = mul(XYZtoRGBMatrix, xyz);
		for (int i = 0; i < 3; i++) {
			// clip to [0,1]
			rgb[i] > 1.0f ? 1.0f : (rgb[i] < 0.0f ? 0.0f : rgb[i]);
			// srgb transform
			rgb[i] = rgb[i] <= 0.0031308f ? rgb[i] * 12.92f : 1.055f * std::pow(rgb[i], (1.0f / 2.4f)) - 0.055f;
		}
		return rgb;
	}

private:
	float c[numSpectrumSamples];

};

Spectrum operator*(const float& f, const Spectrum& s) {
	return s * f;
}


const Spectrum whiteSpectrum{&whiteSpectrumVals};
const Spectrum redSpectrum{&redSpectrumVals};
const Spectrum greenSpectrum{&greenSpectrumVals};
const Spectrum blueSpectrum{&blueSpectrumVals};
const Spectrum cyanSpectrum{&cyanSpectrumVals};
const Spectrum magentaSpectrum{&magentaSpectrumVals};
const Spectrum yellowSpectrum{&yellowSpectrumVals};

Spectrum RGBToSpectrum(const float3 rgb) {
	Spectrum ret{0.0f};
	const float red = rgb[0], green = rgb[1], blue = rgb[2];
	if (red <= green && red <= blue) {
		ret += red * whiteSpectrum;
		if (green <= blue) {
			ret += (green - red) * cyanSpectrum;
			ret += (blue - green) * blueSpectrum;
		} else {
			ret += (blue - red) * cyanSpectrum;
			ret += (green - blue) * greenSpectrum;
		}
	} else if (green <= red && green <= blue) {
		ret += green * whiteSpectrum;
		if (red <= blue) {
			ret += (red - green) * magentaSpectrum;
			ret += (blue - red) * blueSpectrum;
		} else {
			ret += (blue - green) * magentaSpectrum;
			ret += (red - blue) * redSpectrum;
		}
	} else if (blue <= red && blue <= green) {
		ret += blue * whiteSpectrum;
		if (red <= green) {
			ret += (red - blue) * yellowSpectrum;
			ret += (green - red) * greenSpectrum;
		} else {
			ret += (green - blue) * yellowSpectrum;
			ret += (red - green) * redSpectrum;
		}
	}
	return ret;
}


class PointLightSource {
public:
	float3 position, wattage;
};



class Ray {
public:
	float3 o, d, rxOrigin, ryOrigin, rxDir, ryDir, dpdx, dpdy, dddx, dddy;
	bool hasDifferentials = false;
	Ray() : o(), d(float3(0.0f, 0.0f, 1.0f)) {}
	Ray(const float3& o, const float3& d) : o(o), d(d) {}
};



// uber material
// "type" will tell the actual type
// ====== implement it in A2, if you want ======
enum enumMaterialType {
	MAT_LAMBERTIAN,
	MAT_METAL,
	MAT_GLASS
};
class Material {
public:
	std::string name;

	enumMaterialType type = MAT_LAMBERTIAN;
	float eta = 1.0f;
	float glossiness = 1.0f;
	float roughness = 0.0f; // for microfacet model

	float3 baseColor = float3(0.0f); // for metals
	float3 Ka = float3(0.0f);
	float3 Kd = float3(0.9f);
	float3 Ks = float3(0.0f);
	float Ns = 0.0;

	// support 8-bit texture
	bool isTextured = false;
	unsigned char* texture = nullptr;
	unsigned char** mipmap = nullptr;
	int numMaps;
	int textureWidth = 0;
	int textureHeight = 0;

	void generateMipmap() {
		if (texture == nullptr) {
			return;
		}
		std::cout << "init " << name << " mipmap" << std::endl;
		numMaps = floor(std::max(log2(textureWidth), log2(textureHeight)));
		mipmap = new unsigned char*[numMaps];
		std::cout << "has " << numMaps << " maps" << std::endl;

		int width = std::max(1, textureWidth / 2);
		int height = std::max(1, textureHeight / 2);

		int level = 0;
		for (int i = 0; i < numMaps; i++) {
			unsigned char* minimizedImage = new unsigned char[width * height * 3];
			stbir_resize_uint8(texture, textureWidth, textureHeight, 0, minimizedImage, width, height, 0, 3);
			mipmap[level] = minimizedImage;
			// std::cout << level << " " << width << " " << height << std::endl;
			if (width == 1 && height == 1) {
				numMaps = i + 1;
				break;
			}
			width = std::max(1, width / 2);
			height = std::max(1, height / 2);
			level++;
		}
	}

	Material() {};
	virtual ~Material() {};

	void setReflectance(const float3& c) {
		if (type == MAT_LAMBERTIAN) {
			Kd = c;
		} else if (type == MAT_METAL) {
			// empty
		} else if (type == MAT_GLASS) {
			// empty
		}
	}

	float3 fetchTexture(const float2& tex, float mipVal = 0.0f) const {
		int mipLevel = int(floor(mipVal));
		assert(mipLevel <= numMaps);
		float3 mip1;
		for (auto& lvl : {mipLevel, mipLevel + 1}) {
			int width = textureWidth;
			int height = textureHeight;
			const unsigned char* texMap = texture;
			if (lvl > 0) {
				texMap = mipmap[lvl - 1];
				for (int l = lvl; l > 0; l--) {
					width = std::floor(width / 2);
					height = std::floor(height / 2);
				}
			}
			// repeating
			int2x4 bilinCoords = {
				{int(floor(tex.x * width)) % width, int(floor(tex.y * height)) % height}, // u00
				{int(ceil(tex.x * width)) % width, int(floor(tex.y * height)) % height}, // u10
				{int(floor(tex.x * width)) % width, int(ceil(tex.y * height)) % height}, // u01
				{int(ceil(tex.x * width)) % width, int(ceil(tex.y * height)) % height} // u11
			};
			float3x4 bilinPixels;
			float s = tex.x * width - floor(tex.x * width);
			float t = tex.y * height - floor(tex.y * height);
			for (int i = 0; i < 4; i++) {
				if (bilinCoords[i].x < 0) bilinCoords[i].x += width;
				if (bilinCoords[i].y < 0) bilinCoords[i].y += height;
				int pix = (bilinCoords[i].x + bilinCoords[i].y * width) * 3;
				bilinPixels[i][0] = texMap[pix + 0];
				bilinPixels[i][1] = texMap[pix + 1];
				bilinPixels[i][2] = texMap[pix + 2];
			}

			float3 u0 = lerp(bilinPixels[0], bilinPixels[1], s);
			float3 u1 = lerp(bilinPixels[2], bilinPixels[3], s);

			if (lvl == mipLevel + 1) return lerp(mip1, lerp(u0, u1, t), mipVal - float(mipLevel)) / 255.0f;

			mip1 = lerp(u0, u1, t);
			if (mipLevel == numMaps - 1 || mipVal - float(mipLevel) < Epsilon) return mip1 / 255.0f;
		}

		// int x = int(tex.x * width) % width;
		// int y = int(tex.y * height) % height;
		// if (x < 0) x += width;
		// if (y < 0) y += height;

		// int pix = (x + y * width) * 3;
		// unsigned char r = texMap[pix + 0];
		// unsigned char g = texMap[pix + 1];
		// unsigned char b = texMap[pix + 2];

		// if (mipLevel == 0) {
		// 	r = 255;
		// 	g = 0;
		// 	b = 0;
		// } else if (mipLevel == 1) {
		// 	r = 0;
		// 	g = 255;
		// 	b = 0;
		// } else if (mipLevel == 2) {
		// 	r = 0;
		// 	g = 0;
		// 	b = 255;
		// } else if (mipLevel == 3) {
		// 	r = 255;
		// 	g = 255;
		// 	b = 0;
		// } else if (mipLevel == 4) {
		// 	r = 0;
		// 	g = 255;
		// 	b = 255;
		// } else if (mipLevel == 5) {
		// 	r = 255;
		// 	g = 0;
		// 	b = 255;
		// } else if (mipLevel > 5) {
		// 	r = 255;
		// 	g = 255;
		// 	b = 255;
		// }
		// return lerp(u0, u1, t) / 255.0f;
	}

	float3 FSchlick(float3 F0, float cosTheta) const {
		return F0 + (1.0f - F0) * pow(1.0f - cosTheta, 5.0f);
	}

	float microfacetNormalDist(float alpha, float cosTheta) const {
		return pow(alpha, 2.0f) / (PI * pow(pow(cosTheta, 2.0f) * (pow(alpha, 2.0f) - 1) + 1, 2.0f));
	}

	float G1SchlickGGX(float alpha, float cosTheta) const {
		return std::max(cosTheta, 0.001f) / (cosTheta * (1 - alpha / 2) + alpha / 2);
	}

	float GSmith(float3 n, float3 wi, float3 wo, float alpha) const {
		return G1SchlickGGX(alpha, dot(n, wi)) * G1SchlickGGX(alpha, dot(n, wo));
	}

	float3 microFacetBRDF(const float3& lightDir, const float3& viewDir, const float3& N) const {
		float3 h = normalize(lightDir + viewDir);
		float alpha = pow(roughness, 2.0f);
		float3 F0;
		int metallic = 0;

		if (type == MAT_METAL) {
			metallic = 1;
			F0 = baseColor;
		} else {
			F0 = float3(0.16f * pow(Kd, 2.0f));
		}

		float3 F = FSchlick(F0, dot(viewDir, h));
		float D = microfacetNormalDist(alpha, dot(N, h));
		float G = GSmith(N, lightDir, viewDir, alpha);

		float3 spec = (F * D * G) / (4.0f * std::max(dot(N, lightDir), 0.001f) * std::max(dot(N, viewDir), 0.001f));
		/*if (type == MAT_LAMBERTIAN){
			std::cout << spec[0] << spec[1] << spec[2] << std::endl;
		}*/
		return (1 - metallic) * (Kd / PI) + spec;
	}

	float3 BRDF(const float3& wi, const float3& wo, const float3& n) const {
		float3 brdfValue = float3(0.0f);
		if (type == MAT_LAMBERTIAN) {
			// BRDF
			brdfValue = Kd / PI;
		} else if (type == MAT_METAL) {
			// empty
		} else if (type == MAT_GLASS) {
			// empty
		}
		return brdfValue;
	};

	float PDF(const float3& wGiven, const float3& wSample) const {
		// probability density function for a given direction and a given sample
		// it has to be consistent with the sampler
		float pdfValue = 0.0f;
		if (type == MAT_LAMBERTIAN) {
			// empty
		} else if (type == MAT_METAL) {
			// empty
		} else if (type == MAT_GLASS) {
			// empty
		}
		return pdfValue;
	}

	float3 sampler(const float3& wGiven, float& pdfValue) const {
		// sample a vector and record its probability density as pdfValue
		float3 smp = float3(0.0f);
		if (type == MAT_LAMBERTIAN) {
			// empty
		} else if (type == MAT_METAL) {
			// empty
		} else if (type == MAT_GLASS) {
			// empty
		}

		pdfValue = PDF(wGiven, smp);
		return smp;
	}
};





class HitInfo {
public:
	float t; // distance
	float3 P; // location
	float3 N; // shading normal vector
	float2 T; // texture coordinate
	float3 geoN; // geometric normal
	float3 dpdx, dpdy, dddx, dddy, dNdx, dNdy;
	float dudx, dvdx, dudy, dvdy;
	bool isLight = false;
	float3 intensity;
	const Material* material; // const pointer to the material of the intersected object
};



// axis-aligned bounding box
class AABB {
private:
	float3 minp, maxp, size;

public:
	float3 get_minp() const { return minp; };
	float3 get_maxp() const { return maxp; };
	float3 get_size() { return size; };
	int count;


	AABB() {
		minp = float3(FLT_MAX);
		maxp = float3(-FLT_MAX);
		size = float3(0.0f);
		count = 0;
	}

	void reset() {
		minp = float3(FLT_MAX);
		maxp = float3(-FLT_MAX);
		size = float3(0.0f);
		count = 0;
	}

	int getLargestAxis() const {
		if ((size.x > size.y) && (size.x > size.z)) {
			return 0;
		} else if (size.y > size.z) {
			return 1;
		} else {
			return 2;
		}
	}

	void fit(const float3& v) {
		if (minp.x > v.x) minp.x = v.x;
		if (minp.y > v.y) minp.y = v.y;
		if (minp.z > v.z) minp.z = v.z;

		if (maxp.x < v.x) maxp.x = v.x;
		if (maxp.y < v.y) maxp.y = v.y;
		if (maxp.z < v.z) maxp.z = v.z;

		size = maxp - minp;
	}

	float area() const {
		return (2.0f * (size.x * size.y + size.y * size.z + size.z * size.x));
	}


	bool intersect(HitInfo& minHit, const Ray& ray) const {
		// set minHit.t as the distance to the intersection point
		// return true/false if the ray hits or not
		float tx1 = (minp.x - ray.o.x) / ray.d.x;
		float ty1 = (minp.y - ray.o.y) / ray.d.y;
		float tz1 = (minp.z - ray.o.z) / ray.d.z;

		float tx2 = (maxp.x - ray.o.x) / ray.d.x;
		float ty2 = (maxp.y - ray.o.y) / ray.d.y;
		float tz2 = (maxp.z - ray.o.z) / ray.d.z;

		if (tx1 > tx2) {
			const float temp = tx1;
			tx1 = tx2;
			tx2 = temp;
		}

		if (ty1 > ty2) {
			const float temp = ty1;
			ty1 = ty2;
			ty2 = temp;
		}

		if (tz1 > tz2) {
			const float temp = tz1;
			tz1 = tz2;
			tz2 = temp;
		}

		float t1 = tx1; if (t1 < ty1) t1 = ty1; if (t1 < tz1) t1 = tz1;
		float t2 = tx2; if (t2 > ty2) t2 = ty2; if (t2 > tz2) t2 = tz2;

		if (t1 > t2) return false;
		if ((t1 < 0.0) && (t2 < 0.0)) return false;

		minHit.t = t1;
		return true;
	}
};




// triangle
struct Triangle {
	float3 positions[3];
	float3 normals[3];
	float2 texcoords[3];
	int idMaterial = 0;
	AABB bbox;
	float3 center;
	int particleId = -1;
};



// triangle mesh
static float3 shade(const HitInfo& hit, const float3& viewDir, const int level = 0);
class TriangleMesh {
public:
	std::vector<Triangle> triangles;
	std::vector<Material> materials;
	float area = 0.0f;
	AABB bbox;

	void transform(const float4x4& m) {
		// ====== implement it if you want =====
		// matrix transformation of an object	
		// m is a matrix that transforms an object
		// implement proper transformation for positions and normals
		// (hint: you will need to have float4 versions of p and n)
		for (unsigned int i = 0; i < this->triangles.size(); i++) {
			for (int k = 0; k <= 2; k++) {
				const float3 &p = this->triangles[i].positions[k];
				const float3 &n = this->triangles[i].normals[k];
				// not doing anything right now
				float4 p4 = { p[0], p[1], p[2], 1 };
				float4 n4 = { n[0], n[1], n[2], 0 };

				p4 = mul(m, p4);
				n4 = mul(transpose(inverse(m)), n4);

				//printf("before %f %f %f\n", n[0], n[1], n[2]);
				//printf("after %f %f %f\n", n4[0], n4[1], n4[2]);
				
				this->triangles[i].positions[k][0] = p4[0];
				this->triangles[i].positions[k][1] = p4[1];
				this->triangles[i].positions[k][2] = p4[2];

				this->triangles[i].normals[k][0] = n4[0];
				this->triangles[i].normals[k][1] = n4[1];
				this->triangles[i].normals[k][2] = n4[2];
				this->triangles[i].normals[k] = normalize(this->triangles[i].normals[k]);
			}
			//const float3 e0 = this->triangles[i].positions[1] - this->triangles[i].positions[0];
			//const float3 e1 = this->triangles[i].positions[2] - this->triangles[i].positions[0];
			//this->triangles[i].normals[0] = normalize(cross(e0, e1));
			//this->triangles[i].normals[1] = normalize(cross(e0, e1));
			//this->triangles[i].normals[2] = normalize(cross(e0, e1));
		}
	}

	std::tuple<float3, float3, int> samplePoint() {
		int ind = rand() % triangles.size();
		Triangle triSample = triangles[ind];
		float u1 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		float u2 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		float alpha = 1.0f - std::sqrt(u1);
		float beta = (1.0f - u2) * std::sqrt(u1);
		float gamma = u2 * std::sqrt(u1);
		return std::tuple(alpha * triSample.positions[0] + beta * triSample.positions[2] + gamma * triSample.positions[1],
			alpha * triSample.normals[0] + beta * triSample.normals[2] + gamma * triSample.normals[1],
			ind);
	}

	float3 getCenter() {
		return (bbox.get_minp() + bbox.get_maxp()) / 2.0f;
	}

	static bool onEdge(const float x, const float y, const float x0, const float y0, const float x1, const float y1) {
		// Returns true if the point (x, y) lies on the edge defined by the (x1 - x0, y1 - y0)
		// vector, and if the edge is a left or top edge.
		// Returns false otherwise.
		return (-(x - x0) * (y1 - y0) + (y - y0) * (x1 - x0) == 0) && (-(y1 - y0) < 0 || ((y1 - y0) == 0 && (x1 - x0) > 0));
	}

	static std::pair<float, bool> insideWhichEdgeSide(const float x, const float y, const float x0, const float y0, const float x1, const float y1) {
		// Returns the test value for which side of the line from (x0, y0) to (x1, y1) the poinst (x, y) is on
		return { -(x - x0) * (y1 - y0) + (y - y0) * (x1 - x0), onEdge(x, y, x0, y0, x1, y1) };
	}

	static bool insideTriangle(const float x, const float y, float2x3 points) {
		std::pair<float, bool> a = insideWhichEdgeSide(x, y, points[0][0], points[0][1], points[1][0], points[1][1]);
		std::pair<float, bool> b = insideWhichEdgeSide(x, y, points[1][0], points[1][1], points[2][0], points[2][1]);
		std::pair<float, bool> c = insideWhichEdgeSide(x, y, points[2][0], points[2][1], points[0][0], points[0][1]);

		return ((a.first < 0 || a.second) && (b.first < 0 || b.second) && (c.first < 0 || c.second)) ||
			((a.first > 0 || a.second) && (b.first > 0 || b.second) && (c.first > 0 || c.second));
	}

	void rasterizeTriangle(const Triangle& tri, const float4x4& plm) const {
		// ====== implement it in A1 ======
		// rasterization of a triangle
		// "plm" should be a matrix that contains perspective projection and the camera matrix
		// you do not need to implement clipping
		// you may call the "shade" function to get the pixel value
		// (you may ignore viewDir for now)
		std::vector<float4> clippedPoints;
		std::vector<float4x3> clippedTriangles;

		int triBbox[2][2] = { {-1, -1}, {-1, -1} };
		float2x3 transformedPoints;
		float3 depths;
		float3 wValues;
		float3 wValues_inv;
		float2x3 textureCoordsP;
		int xyzClipped[3] = { 0, 0, 0 };

		for (int i = 0; i < 3; i++) {
			// apply camera and perspective projection matrices to each vertex
			float4 posVec = { tri.positions[i][0], tri.positions[i][1], tri.positions[i][2], 1.0f };
			float4 k = mul(plm, posVec);
			// w important value for later
			float w = k[3];

			xyzClipped[0] += -(k[0] < -w) + (w < k[0]);
			xyzClipped[1] += -(k[1] < -w) + (w < k[1]);
			xyzClipped[2] += k[2] < -w;
		}

		if (abs(xyzClipped[0]) == 3 || abs(xyzClipped[1]) == 3 || xyzClipped[2] == 3)
		{
			return;
		}

		for (int i = 0; i < 3; i++) {
			// apply camera and perspective projection matrices to each vertex
			float4 posVec = { tri.positions[i][0], tri.positions[i][1], tri.positions[i][2], 1.0f };
			float4 k = mul(plm, posVec);
			// w important value for later
			float w = k[3];

			wValues[i] = w;
			// then do perspective divide by the w value from the resulting point to normalize to ndc
			k = k / w;

			// ndc -> screen
			k = mul(ndcFlattenMatrix, k);

			// keep the raw float results for x,y for rasterization
			transformedPoints[i] = {
				k[0],
				k[1]
			};

			// transform depth from [-1,1] to [0,1] and store it
			depths[i] = (k[2] + 1) / 2;

			// use the rounded-down integer x,y positions for calculating bounding box
			int x = int(floor(k[0]));
			int y = int(floor(k[1]));

			// FOR A1 TASK 2:
			// if (FrameBuffer.valid(x, y)) {
			// 	FrameBuffer.pixel(x, y) = float3(1.0f);
			// }

			if (triBbox[0][0] == -1 && triBbox[0][1] == -1 && triBbox[1][0] == -1 && triBbox[1][1] == -1) {
				triBbox[0][0] = std::min(globalWidth, std::max(0, x));
				triBbox[0][1] = std::min(globalHeight, std::max(0, y));
				triBbox[1][0] = std::max(0, std::min(globalWidth, x));
				triBbox[1][1] = std::max(0, std::min(globalHeight, y));
			}
			else {
				triBbox[0][0] = std::max(0, std::min(x, triBbox[0][0]));
				triBbox[0][1] = std::max(0, std::min(y, triBbox[0][1]));
				triBbox[1][0] = std::min(globalWidth, std::max(x, triBbox[1][0]));
				triBbox[1][1] = std::min(globalHeight, std::max(y, triBbox[1][1]));
			}
		}
		// for perspective correct interpolation
		wValues_inv = {
			1 / wValues[0],
			1 / wValues[1],
			1 / wValues[2]
		};
		textureCoordsP = {
			{
				tri.texcoords[0][0] / wValues[0],
				tri.texcoords[0][1] / wValues[0]
			},
			{
				tri.texcoords[1][0] / wValues[1],
				tri.texcoords[1][1] / wValues[1]
			},
			{
				tri.texcoords[2][0] / wValues[2],
				tri.texcoords[2][1] / wValues[2]
			}
		};

		// for finding phi values for each vertex
		float3x3 R = {
			{ 1, transformedPoints[0][0], transformedPoints[0][1] },
			{ 1, transformedPoints[1][0], transformedPoints[1][1] },
			{ 1, transformedPoints[2][0], transformedPoints[2][1] }
		};
		float3x3 R_inv = inverse(R);

		// rasterize
		for (int i = triBbox[0][0]; i <= triBbox[1][0]; i++) {
			for (int j = triBbox[0][1]; j <= triBbox[1][1]; j++) {
				if (insideTriangle(i + 0.5, j + 0.5, transformedPoints)) {
					if (FrameBuffer.valid(i, j)) {
						// start perspective correct interpolation
						float3 x = { 1.0f, static_cast<float>(i), static_cast<float>(j) };
						float3 phi = mul(R_inv, x);
						const float interpolatedW = phi[0] * wValues_inv[0] + phi[1] * wValues_inv[1] + phi[2] * wValues_inv[2];
						const float interpolatedDepth = (phi[0] * (depths[0] / wValues_inv[0]) + phi[1] * (depths[1] / wValues_inv[1]) + phi[2] * (depths[2] / wValues_inv[2])) / interpolatedW;
						const float2 interpolatedTextureCoordsP = {
							phi[0] * textureCoordsP[0][0] + phi[1] * textureCoordsP[1][0] + phi[2] * textureCoordsP[2][0],
							phi[0] * textureCoordsP[0][1] + phi[1] * textureCoordsP[1][1] + phi[2] * textureCoordsP[2][1]
						};
						const float2 interpolatedTextureCoords = {
							interpolatedTextureCoordsP[0] / interpolatedW,
							interpolatedTextureCoordsP[1] / interpolatedW
						};
						// depth buffering
						if (interpolatedDepth < FrameBuffer.depth(i, j)) {
							FrameBuffer.depth(i, j) = interpolatedDepth;
							HitInfo hi;
							hi.T = interpolatedTextureCoords;
							hi.material = &(materials[tri.idMaterial]);
							hi.N = normalize(phi[0] * tri.normals[0] + phi[1] * tri.normals[1] + phi[2] * tri.normals[2]);
							hi.P = phi[0] * tri.positions[0] + phi[1] * tri.positions[1] + phi[2] * tri.positions[2];

							FrameBuffer.pixel(i, j) = shade(hi, globalViewDir);
						}
					}
				}
			}
		}
	}


	bool raytraceTriangle(HitInfo& result, const Ray& ray, const Triangle& tri, float tMin, float tMax) const {
		// ====== implement it in A2 ======
		// ray-triangle intersection
		// fill in "result" when there is an intersection
		// return true/false if there is an intersection or not
		float3 na = tri.normals[0];
		float3 nb = tri.normals[2];
		float3 nc = tri.normals[1];

		float3 ab = tri.positions[0] - tri.positions[2];
		float3 ac = tri.positions[0] - tri.positions[1];
		float3 ao = tri.positions[0] - ray.o;

		float D = dot(cross(ab, ac), ray.d);

		if (abs(D) < Epsilon) {
			return false;
		}

		float invD = 1 / D;


		float beta = dot(cross(ao, ac), ray.d);
		float gamma = dot(cross(ab, ao), ray.d);
		float t = dot(cross(ab, ac), ao);

		beta = beta * invD;
		gamma = gamma * invD;
		t = t * invD;

		if (!(0 < beta && 0 < gamma && beta + gamma < 1) || 0 >= t) {
			return false;
		}

		if (!(tMin < t && t < tMax)) {
			return false;
		}

		float3 nonNormalizedN = (1 - beta - gamma) * na + beta * nb + gamma * nc;
		result.t = t;
		result.N = normalize(nonNormalizedN);
		result.P = ray.o + t * ray.d;
		result.T = {
			(1 - beta - gamma) * tri.texcoords[0][0] + beta * tri.texcoords[2][0] + gamma * tri.texcoords[1][0],
			(1 - beta - gamma) * tri.texcoords[0][1] + beta * tri.texcoords[2][1] + gamma * tri.texcoords[1][1]
		};
		result.material = &(materials[tri.idMaterial]);

		if (ray.hasDifferentials) {
			float3 diffRight = normalize(cross(-globalViewDir, globalUp));
			float tx = -dot(result.N, ray.rxOrigin) / dot(ray.rxDir, result.N);
			float ty = -dot(result.N, ray.ryOrigin) / dot(ray.ryDir, result.N);
			float3 dpdx = ray.dpdx;
			float3 dpdy = ray.dpdy;
			float3 dddx = ray.dddx;
			float3 dddy = ray.dddy;
			float dtdx = -dot((dpdx + tx * dddx), result.N) / dot(ray.rxDir, result.N);
			float dtdy = -dot((dpdy + ty * dddy), result.N) / dot(ray.ryDir, result.N);

			float3 dpdx1 = (dpdx + tx * dddx) + dtdx * ray.rxDir;
			float3 dpdy1 = (dpdy + ty * dddy) + dtdy * ray.ryDir;
			result.dpdx = dpdx1;
			result.dpdy = dpdy1;
			result.dddx = dddx;
			result.dddy = dddy;

			// derivs of barycentrics
			float db0x = sum((1 - beta - gamma) * dpdx1);
			float db1x = sum(beta * dpdx1);
			float db2x = -db0x - db1x;
			float db0y = sum((1 - beta - gamma) * dpdy1);
			float db1y = sum(beta * dpdy1);
			float db2y = -db0y - db1y;

			// derivs of normals
			float3 dndx = db0x * na + db1x * nb + db2x * nc;
			float3 dndy = db0y * na + db1y * nb + db2y * nc;
			//std::cout << db0x << " " << db0y << std::endl;
			result.dNdx = (dot(nonNormalizedN, nonNormalizedN) * dndx - dot(nonNormalizedN, dndx) * nonNormalizedN) / pow(dot(nonNormalizedN, nonNormalizedN), 1.5f);
			result.dNdy = (dot(nonNormalizedN, nonNormalizedN) * dndy - dot(nonNormalizedN, dndy) * nonNormalizedN) / pow(dot(nonNormalizedN, nonNormalizedN), 1.5f);


			// derivs of tex coords
			result.dudx = db0x * tri.texcoords[0][0] + db1x * tri.texcoords[2][0] + db2x * tri.texcoords[1][0];
			result.dvdx = db0x * tri.texcoords[0][1] + db1x * tri.texcoords[2][1] + db2x * tri.texcoords[1][1];
			result.dudy = db0y * tri.texcoords[0][0] + db1y * tri.texcoords[2][0] + db2y * tri.texcoords[1][0];
			result.dvdy = db0y * tri.texcoords[0][1] + db1y * tri.texcoords[2][1] + db2y * tri.texcoords[1][1];
		}


		// calculate geometric normal
		if (tri.normals[0] == tri.normals[1] && tri.normals[0] == tri.normals[2]) {
			result.geoN = tri.normals[0];
		} else {
			const float3 e0 = tri.positions[1] - tri.positions[0];
			const float3 e1 = tri.positions[2] - tri.positions[0];
			result.geoN = normalize(cross(e0, e1));
		}

		return true;
	}


	// some precalculation for bounding boxes (you do not need to change it)
	void preCalc() {
		bbox.reset();
		for (int i = 0, _n = (int)triangles.size(); i < _n; i++) {
			this->triangles[i].bbox.reset();
			this->triangles[i].bbox.fit(this->triangles[i].positions[0]);
			this->triangles[i].bbox.fit(this->triangles[i].positions[1]);
			this->triangles[i].bbox.fit(this->triangles[i].positions[2]);

			this->triangles[i].center = (this->triangles[i].positions[0] + this->triangles[i].positions[1] + this->triangles[i].positions[2]) * (1.0f / 3.0f);

			this->bbox.fit(this->triangles[i].positions[0]);
			this->bbox.fit(this->triangles[i].positions[1]);
			this->bbox.fit(this->triangles[i].positions[2]);

			float3 ab = this->triangles[i].positions[1] - this->triangles[i].positions[0];
			float3 ac = this->triangles[i].positions[2] - this->triangles[i].positions[0];
			this->area += 0.5f * length(ab) * length(ac) * std::sqrt(1 - std::pow(dot(normalize(ab), normalize(ac)), 2.0f));
		}
	}


	// load .obj file (you do not need to modify it unless you want to change something)
	bool load(const char* filename, const float4x4& ctm = linalg::identity) {
		int nVertices = 0;
		float* vertices;
		float* normals;
		float* texcoords;
		int nIndices;
		int* indices;
		int* matid = nullptr;

		printf("Loading \"%s\"...\n", filename);
		ParseOBJ(filename, nVertices, &vertices, &normals, &texcoords, nIndices, &indices, &matid);
		if (nVertices == 0) return false;
		this->triangles.resize(nIndices / 3);

		if (matid != nullptr) {
			for (unsigned int i = 0; i < materials.size(); i++) {
				// convert .mlt data into BSDF definitions
				// you may change the followings in the final project if you want
				materials[i].type = MAT_LAMBERTIAN;
				materials[i].roughness = 0.0f;
				if (materials[i].Ns == 100.0f) {
					materials[i].type = MAT_METAL;
					materials[i].roughness = 0.0f;
					materials[i].baseColor = float3(0.549f, 0.556f, 0.554f);
				}
				if (materials[i].name.compare(0, 5, "glass", 0, 5) == 0) {
					materials[i].type = MAT_GLASS;
					materials[i].eta = 1.5f;
					materials[i].roughness = 0.0f;
				}
			}
		} else {
			// use default Lambertian
			this->materials.resize(1);
			this->materials[0].roughness = 0.0f;
		}

		for (unsigned int i = 0; i < this->triangles.size(); i++) {
			const int v0 = indices[i * 3 + 0];
			const int v1 = indices[i * 3 + 1];
			const int v2 = indices[i * 3 + 2];

			this->triangles[i].positions[0] = float3(vertices[v0 * 3 + 0], vertices[v0 * 3 + 1], vertices[v0 * 3 + 2]);
			this->triangles[i].positions[1] = float3(vertices[v1 * 3 + 0], vertices[v1 * 3 + 1], vertices[v1 * 3 + 2]);
			this->triangles[i].positions[2] = float3(vertices[v2 * 3 + 0], vertices[v2 * 3 + 1], vertices[v2 * 3 + 2]);

			if (normals != nullptr) {
				this->triangles[i].normals[0] = float3(normals[v0 * 3 + 0], normals[v0 * 3 + 1], normals[v0 * 3 + 2]);
				this->triangles[i].normals[1] = float3(normals[v1 * 3 + 0], normals[v1 * 3 + 1], normals[v1 * 3 + 2]);
				this->triangles[i].normals[2] = float3(normals[v2 * 3 + 0], normals[v2 * 3 + 1], normals[v2 * 3 + 2]);
			} else {
				// no normal data, calculate the normal for a polygon
				const float3 e0 = this->triangles[i].positions[1] - this->triangles[i].positions[0];
				const float3 e1 = this->triangles[i].positions[2] - this->triangles[i].positions[0];
				const float3 n = normalize(cross(e0, e1));

				this->triangles[i].normals[0] = n;
				this->triangles[i].normals[1] = n;
				this->triangles[i].normals[2] = n;
			}

			// material id
			this->triangles[i].idMaterial = 0;
			if (matid != nullptr) {
				// read texture coordinates
				if ((texcoords != nullptr) && materials[matid[i]].isTextured) {
					this->triangles[i].texcoords[0] = float2(texcoords[v0 * 2 + 0], texcoords[v0 * 2 + 1]);
					this->triangles[i].texcoords[1] = float2(texcoords[v1 * 2 + 0], texcoords[v1 * 2 + 1]);
					this->triangles[i].texcoords[2] = float2(texcoords[v2 * 2 + 0], texcoords[v2 * 2 + 1]);
				} else {
					this->triangles[i].texcoords[0] = float2(0.0f);
					this->triangles[i].texcoords[1] = float2(0.0f);
					this->triangles[i].texcoords[2] = float2(0.0f);
				}
				this->triangles[i].idMaterial = matid[i];
			} else {
				this->triangles[i].texcoords[0] = float2(0.0f);
				this->triangles[i].texcoords[1] = float2(0.0f);
				this->triangles[i].texcoords[2] = float2(0.0f);
			}
		}
		printf("Loaded \"%s\" with %d triangles.\n", filename, int(triangles.size()));

		delete[] vertices;
		delete[] normals;
		delete[] texcoords;
		delete[] indices;
		delete[] matid;

		return true;
	}

	~TriangleMesh() {
		for (auto& m : materials) {
			if (m.mipmap != nullptr) {
				if (m.numMaps > 0) {
					for (int i = 0; i < m.numMaps; i++) {
						delete[] m.mipmap[i];
					}
				}
				delete[] m.mipmap;
			}
		}
		materials.clear();
		triangles.clear();
	}


	bool bruteforceIntersect(HitInfo& result, const Ray& ray, float tMin = 0.0f, float tMax = FLT_MAX) {
		// bruteforce ray tracing (for debugging)
		bool hit = false;
		HitInfo tempMinHit;
		result.t = FLT_MAX;

		for (int i = 0; i < triangles.size(); ++i) {
			if (raytraceTriangle(tempMinHit, ray, triangles[i], tMin, tMax)) {
				if (tempMinHit.t < result.t) {
					hit = true;
					result = tempMinHit;
				}
			}
		}

		return hit;
	}

	void createSingleQuad(float3 vertices[4]) {
		// vertices must be in either clockwise or counter-clockwise order, cannot be arbitrary
		int i = triangles.size();
		triangles.resize(i + 2);
		materials.resize(1);


		triangles[i + 0].idMaterial = 0;
		triangles[i + 1].idMaterial = 0;

		// arealight.vertices[0] = float3(0.0f, 0.0f, 2.0f);
		// arealight.vertices[1] = float3(1.0f, 0.0f, 2.0f);
		// arealight.vertices[2] = float3(0.0f, 1.0f, 2.0f);
		// arealight.vertices[3] = float3(1.0f, 1.0f, 2.0f);
		triangles[i + 0].positions[0] = vertices[0];//float3(-2.0f, 0.0f, 0.0f);
		triangles[i + 0].positions[1] = vertices[1];//float3(-2.0f, 1.0f, 0.0f);
		triangles[i + 0].positions[2] = vertices[3];//float3(-2.0f, 0.0f, 1.0f);

		triangles[i + 1].positions[0] = vertices[3];//float3(-2.0f, 0.0f, 1.0f);
		triangles[i + 1].positions[1] = vertices[1];//float3(-2.0f, 1.0f, 0.0f);
		triangles[i + 1].positions[2] = vertices[2];//float3(-2.0f, 1.0f, 1.0f);

		const float3 e00 = this->triangles[i + 0].positions[1] - this->triangles[i + 0].positions[0];
		const float3 e10 = this->triangles[i + 0].positions[2] - this->triangles[i + 0].positions[0];
		const float3 n0 = normalize(cross(e00, e10));
		const float3 e01 = this->triangles[i + 1].positions[1] - this->triangles[i + 0].positions[0];
		const float3 e11 = this->triangles[i + 1].positions[2] - this->triangles[i + 0].positions[0];
		const float3 n1 = normalize(cross(e01, e11));

		triangles[i + 0].normals[0] = n0;
		triangles[i + 0].normals[1] = n0;
		triangles[i + 0].normals[2] = n0;
		triangles[i + 1].normals[0] = n1;
		triangles[i + 1].normals[1] = n1;
		triangles[i + 1].normals[2] = n1;

		triangles[i + 0].texcoords[0] = float2(0.0f, 0.0f);
		triangles[i + 0].texcoords[1] = float2(0.0f, 1.0f);
		triangles[i + 0].texcoords[2] = float2(1.0f, 0.0f);
		triangles[i + 1].texcoords[0] = float2(0.0f, 0.0f);
		triangles[i + 1].texcoords[1] = float2(0.0f, 1.0f);
		triangles[i + 1].texcoords[2] = float2(1.0f, 0.0f);
	}

	void createSingleTriangle() {
		triangles.resize(1);
		materials.resize(1);

		triangles[0].idMaterial = 0;

		triangles[0].positions[0] = float3(-0.5f, -0.5f, 0.0f);
		triangles[0].positions[1] = float3(0.5f, -0.5f, 0.0f);
		triangles[0].positions[2] = float3(0.0f, 0.5f, 0.0f);

		const float3 e0 = this->triangles[0].positions[1] - this->triangles[0].positions[0];
		const float3 e1 = this->triangles[0].positions[2] - this->triangles[0].positions[0];
		const float3 n = normalize(cross(e0, e1));

		triangles[0].normals[0] = n;
		triangles[0].normals[1] = n;
		triangles[0].normals[2] = n;

		triangles[0].texcoords[0] = float2(0.0f, 0.0f);
		triangles[0].texcoords[1] = float2(0.0f, 1.0f);
		triangles[0].texcoords[2] = float2(1.0f, 0.0f);
	}


private:
	// === you do not need to modify the followings in this class ===
	void loadTexture(const char* fname, const int i) {
		int comp;
		materials[i].texture = stbi_load(fname, &materials[i].textureWidth, &materials[i].textureHeight, &comp, 3);
		if (!materials[i].texture) {
			std::cerr << "Unable to load texture: " << fname << std::endl;
			return;
		}
		materials[i].generateMipmap();
	}

	std::string GetBaseDir(const std::string& filepath) {
		if (filepath.find_last_of("/\\") != std::string::npos) return filepath.substr(0, filepath.find_last_of("/\\"));
		return "";
	}
	std::string base_dir;

	void LoadMTL(const std::string fileName) {
		FILE* fp = fopen(fileName.c_str(), "r");

		Material mtl;
		mtl.texture = nullptr;
		char line[81];
		while (fgets(line, 80, fp) != nullptr) {
			float r, g, b, s;
			std::string lineStr;
			lineStr = line;
			int i = int(materials.size());

			if (lineStr.compare(0, 6, "newmtl", 0, 6) == 0) {
				lineStr.erase(0, 7);
				mtl.name = lineStr;
				mtl.isTextured = false;
			} else if (lineStr.compare(0, 2, "Ka", 0, 2) == 0) {
				lineStr.erase(0, 3);
				sscanf(lineStr.c_str(), "%f %f %f\n", &r, &g, &b);
				mtl.Ka = float3(r, g, b);

				// float3 col = mtl.Ka;
				// printf("raw color: %f, %f, %f\n", col[0], col[1], col[2]);
				// col = Spectrum(RGBToSpectrum(col)).convertToRGB();
				// printf("converted color: %f, %f, %f\n", col[0], col[1], col[2]);
				// float3 red = redSpectrum.convertToRGB();
				// printf("red: %f, %f, %f\n", red[0], red[1], red[2]);
			} else if (lineStr.compare(0, 2, "Kd", 0, 2) == 0) {
				lineStr.erase(0, 3);
				sscanf(lineStr.c_str(), "%f %f %f\n", &r, &g, &b);
				mtl.Kd = float3(r, g, b);
			} else if (lineStr.compare(0, 2, "Ks", 0, 2) == 0) {
				lineStr.erase(0, 3);
				sscanf(lineStr.c_str(), "%f %f %f\n", &r, &g, &b);
				mtl.Ks = float3(r, g, b);
			} else if (lineStr.compare(0, 2, "Ns", 0, 2) == 0) {
				lineStr.erase(0, 3);
				sscanf(lineStr.c_str(), "%f\n", &s);
				mtl.Ns = s;
				mtl.texture = nullptr;
				materials.push_back(mtl);
			} else if (lineStr.compare(0, 6, "map_Kd", 0, 6) == 0) {
				lineStr.erase(0, 7);
				lineStr.erase(lineStr.size() - 1, 1);
				materials[i - 1].isTextured = true;
				loadTexture((base_dir + lineStr).c_str(), i - 1);
			}
		}

		fclose(fp);
	}

	void ParseOBJ(const char* fileName, int& nVertices, float** vertices, float** normals, float** texcoords, int& nIndices, int** indices, int** materialids) {
		// local function in C++...
		struct {
			void operator()(char* word, int* vindex, int* tindex, int* nindex) {
				const char* null = " ";
				char* ptr;
				const char* tp;
				const char* np;

				// by default, the texture and normal pointers are set to the null string
				tp = null;
				np = null;

				// replace slashes with null characters and cause tp and np to point
				// to character immediately following the first or second slash
				for (ptr = word; *ptr != '\0'; ptr++) {
					if (*ptr == '/') {
						if (tp == null) {
							tp = ptr + 1;
						} else {
							np = ptr + 1;
						}

						*ptr = '\0';
					}
				}

				*vindex = atoi(word);
				*tindex = atoi(tp);
				*nindex = atoi(np);
			}
		} get_indices;

		base_dir = GetBaseDir(fileName);
		#ifdef _WIN32
			base_dir += "\\";
		#else
			base_dir += "/";
		#endif

		FILE* fp = fopen(fileName, "r");
		int nv = 0, nn = 0, nf = 0, nt = 0;
		char line[81];
		if (!fp) {
			printf("Cannot open \"%s\" for reading\n", fileName);
			return;
		}

		while (fgets(line, 80, fp) != NULL) {
			std::string lineStr;
			lineStr = line;

			if (lineStr.compare(0, 6, "mtllib", 0, 6) == 0) {
				lineStr.erase(0, 7);
				lineStr.erase(lineStr.size() - 1, 1);
				LoadMTL(base_dir + lineStr);
			}

			if (line[0] == 'v') {
				if (line[1] == 'n') {
					nn++;
				} else if (line[1] == 't') {
					nt++;
				} else {
					nv++;
				}
			} else if (line[0] == 'f') {
				nf++;
			}
		}
		fseek(fp, 0, 0);

		float* n = new float[3 * (nn > nf ? nn : nf)];
		float* v = new float[3 * nv];
		float* t = new float[2 * nt];

		int* vInd = new int[3 * nf];
		int* nInd = new int[3 * nf];
		int* tInd = new int[3 * nf];
		int* mInd = new int[nf];

		int nvertices = 0;
		int nnormals = 0;
		int ntexcoords = 0;
		int nindices = 0;
		int ntriangles = 0;
		bool noNormals = false;
		bool noTexCoords = false;
		bool noMaterials = true;
		int cmaterial = 0;

		while (fgets(line, 80, fp) != NULL) {
			std::string lineStr;
			lineStr = line;

			if (line[0] == 'v') {
				if (line[1] == 'n') {
					float x, y, z;
					sscanf(&line[2], "%f %f %f\n", &x, &y, &z);
					float l = sqrt(x * x + y * y + z * z);
					x = x / l;
					y = y / l;
					z = z / l;
					n[nnormals] = x;
					nnormals++;
					n[nnormals] = y;
					nnormals++;
					n[nnormals] = z;
					nnormals++;
				} else if (line[1] == 't') {
					float u, v;
					sscanf(&line[2], "%f %f\n", &u, &v);
					t[ntexcoords] = u;
					ntexcoords++;
					t[ntexcoords] = v;
					ntexcoords++;
				} else {
					float x, y, z;
					sscanf(&line[1], "%f %f %f\n", &x, &y, &z);
					v[nvertices] = x;
					nvertices++;
					v[nvertices] = y;
					nvertices++;
					v[nvertices] = z;
					nvertices++;
				}
			}
			if (lineStr.compare(0, 6, "usemtl", 0, 6) == 0) {
				lineStr.erase(0, 7);
				if (materials.size() != 0) {
					for (unsigned int i = 0; i < materials.size(); i++) {
						if (lineStr.compare(materials[i].name) == 0) {
							cmaterial = i;
							noMaterials = false;
							break;
						}
					}
				}

			} else if (line[0] == 'f') {
				char s1[32], s2[32], s3[32];
				int vI, tI, nI;
				sscanf(&line[1], "%s %s %s\n", s1, s2, s3);

				mInd[ntriangles] = cmaterial;

				// indices for first vertex
				get_indices(s1, &vI, &tI, &nI);
				vInd[nindices] = vI - 1;
				if (nI) {
					nInd[nindices] = nI - 1;
				} else {
					noNormals = true;
				}

				if (tI) {
					tInd[nindices] = tI - 1;
				} else {
					noTexCoords = true;
				}
				nindices++;

				// indices for second vertex
				get_indices(s2, &vI, &tI, &nI);
				vInd[nindices] = vI - 1;
				if (nI) {
					nInd[nindices] = nI - 1;
				} else {
					noNormals = true;
				}

				if (tI) {
					tInd[nindices] = tI - 1;
				} else {
					noTexCoords = true;
				}
				nindices++;

				// indices for third vertex
				get_indices(s3, &vI, &tI, &nI);
				vInd[nindices] = vI - 1;
				if (nI) {
					nInd[nindices] = nI - 1;
				} else {
					noNormals = true;
				}

				if (tI) {
					tInd[nindices] = tI - 1;
				} else {
					noTexCoords = true;
				}
				nindices++;

				ntriangles++;
			}
		}

		*vertices = new float[ntriangles * 9];
		if (!noNormals) {
			*normals = new float[ntriangles * 9];
		} else {
			*normals = 0;
		}

		if (!noTexCoords) {
			*texcoords = new float[ntriangles * 6];
		} else {
			*texcoords = 0;
		}

		if (!noMaterials) {
			*materialids = new int[ntriangles];
		} else {
			*materialids = 0;
		}

		*indices = new int[ntriangles * 3];
		nVertices = ntriangles * 3;
		nIndices = ntriangles * 3;

		for (int i = 0; i < ntriangles; i++) {
			if (!noMaterials) {
				(*materialids)[i] = mInd[i];
			}

			(*indices)[3 * i] = 3 * i;
			(*indices)[3 * i + 1] = 3 * i + 1;
			(*indices)[3 * i + 2] = 3 * i + 2;

			(*vertices)[9 * i] = v[3 * vInd[3 * i]];
			(*vertices)[9 * i + 1] = v[3 * vInd[3 * i] + 1];
			(*vertices)[9 * i + 2] = v[3 * vInd[3 * i] + 2];

			(*vertices)[9 * i + 3] = v[3 * vInd[3 * i + 1]];
			(*vertices)[9 * i + 4] = v[3 * vInd[3 * i + 1] + 1];
			(*vertices)[9 * i + 5] = v[3 * vInd[3 * i + 1] + 2];

			(*vertices)[9 * i + 6] = v[3 * vInd[3 * i + 2]];
			(*vertices)[9 * i + 7] = v[3 * vInd[3 * i + 2] + 1];
			(*vertices)[9 * i + 8] = v[3 * vInd[3 * i + 2] + 2];

			if (!noNormals) {
				(*normals)[9 * i] = n[3 * nInd[3 * i]];
				(*normals)[9 * i + 1] = n[3 * nInd[3 * i] + 1];
				(*normals)[9 * i + 2] = n[3 * nInd[3 * i] + 2];

				(*normals)[9 * i + 3] = n[3 * nInd[3 * i + 1]];
				(*normals)[9 * i + 4] = n[3 * nInd[3 * i + 1] + 1];
				(*normals)[9 * i + 5] = n[3 * nInd[3 * i + 1] + 2];

				(*normals)[9 * i + 6] = n[3 * nInd[3 * i + 2]];
				(*normals)[9 * i + 7] = n[3 * nInd[3 * i + 2] + 1];
				(*normals)[9 * i + 8] = n[3 * nInd[3 * i + 2] + 2];
			}

			if (!noTexCoords) {
				(*texcoords)[6 * i] = t[2 * tInd[3 * i]];
				(*texcoords)[6 * i + 1] = t[2 * tInd[3 * i] + 1];

				(*texcoords)[6 * i + 2] = t[2 * tInd[3 * i + 1]];
				(*texcoords)[6 * i + 3] = t[2 * tInd[3 * i + 1] + 1];

				(*texcoords)[6 * i + 4] = t[2 * tInd[3 * i + 2]];
				(*texcoords)[6 * i + 5] = t[2 * tInd[3 * i + 2] + 1];
			}

		}
		fclose(fp);

		delete[] n;
		delete[] v;
		delete[] t;
		delete[] nInd;
		delete[] vInd;
		delete[] tInd;
		delete[] mInd;
	}
};



// BVH node (for A2 extra)
class BVHNode {
public:
	bool isLeaf;
	int idLeft, idRight;
	int triListNum;
	int* triList;
	AABB bbox;
};

class BVHSplit {
public:
	bool isStart;
	float pos;
	const Triangle* obj;
	int obj_ind;
	int* sorted_ind;
};


// ====== implement it in A2 extra ======
// fill in the missing parts
class BVH {
public:
	const TriangleMesh* triangleMesh = nullptr;
	BVHNode* node = nullptr;

	const float costBBox = 1.0f;
	const float costTri = 1.0f;

	int leafNum = 0;
	int nodeNum = 0;
	int levels = 0;

	BVH() {}
	void build(const TriangleMesh* mesh);

	bool intersect(HitInfo& result, const Ray& ray, float tMin = 0.0f, float tMax = FLT_MAX, int ignoreParticleId = -1) const {
		bool hit = false;
		HitInfo tempMinHit;
		result.t = FLT_MAX;

		// bvh
		if (this->node[0].bbox.intersect(tempMinHit, ray)) {
			hit = traverse(result, ray, 0, tMin, tMax, ignoreParticleId);
		}
		if (result.t != FLT_MAX) hit = true;

		return hit;
	}
	bool traverse(HitInfo& result, const Ray& ray, int node_id, float tMin, float tMax, int ignoreParticleId) const;

private:
	float getPosAxis(const int obj_ind, const char axis, const char mode) const;
	void sortAxis(int* obj_index, const char axis, const int li, const int ri, const char mode) const;
	int splitBVH(int* obj_index, const int obj_num, const AABB& bbox, int level = 0);

};

float BVH::getPosAxis(const int obj_ind, const char axis, const char mode) const
{
	Triangle tri = triangleMesh->triangles[obj_ind];
	float val = tri.center[axis];
	if (mode == 'l') {
		val = tri.bbox.get_minp()[axis];
	} else if (mode == 'r') {
		val = tri.bbox.get_maxp()[axis];
	}
	return val;
}

// sort bounding boxes (in case you want to build SAH-BVH)
void BVH::sortAxis(int* obj_index, const char axis, const int li, const int ri, const char mode = 'c') const {
	int i, j;
	float pivot;
	int temp;

	i = li;
	j = ri;

	pivot = getPosAxis(obj_index[(li + ri) / 2], axis, mode);

	while (true) {
		while (getPosAxis(obj_index[i], axis, mode) < pivot) {
			++i;
		}

		while (getPosAxis(obj_index[j], axis, mode) > pivot) {
			--j;
		}

		if (i >= j) break;

		temp = obj_index[i];
		obj_index[i] = obj_index[j];
		obj_index[j] = temp;

		++i;
		--j;
	}

	if (li < (i - 1)) sortAxis(obj_index, axis, li, i - 1, mode);
	if ((j + 1) < ri) sortAxis(obj_index, axis, j + 1, ri, mode);
}

void print_arr(std::vector<int>* arr, int size) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < arr[i].size(); j++) {
			std::cout << arr[i][j] << " ";
		}
		std::cout << std::endl;
	}
}

void print_arr1(int* arr, int size) {
	for (int i = 0; i < size; i++) {
		std::cout << arr[i] << " ";
	}
	std::cout << std::endl;
}


//#define SAHBVH // use this in once you have SAH-BVH
int BVH::splitBVH(int* obj_index, const int obj_num, const AABB& bbox, int level) {
	// ====== exntend it in A2 extra ======
#ifndef SAHBVH
	int bestAxis, bestIndex;
	AABB bboxL, bboxR, bestbboxL, bestbboxR;
	int* sorted_obj_index[3];
	sorted_obj_index[0] = new int[obj_num];
	sorted_obj_index[1] = new int[obj_num];
	sorted_obj_index[2] = new int[obj_num];

	// split along the largest axis
	bestAxis = bbox.getLargestAxis();

	// sorting along the axis
	this->sortAxis(obj_index, bestAxis, 0, obj_num - 1);
	for (int i = 0; i < obj_num; ++i) {
		sorted_obj_index[bestAxis][i] = obj_index[i];
	}

	// split in the middle
	bestIndex = obj_num / 2 - 1;

	bboxL.reset();
	for (int i = 0; i <= bestIndex; ++i) {
		const Triangle& tri = triangleMesh->triangles[obj_index[i]];
		bboxL.fit(tri.positions[0]);
		bboxL.fit(tri.positions[1]);
		bboxL.fit(tri.positions[2]);
	}

	bboxR.reset();
	for (int i = bestIndex + 1; i < obj_num; ++i) {
		const Triangle& tri = triangleMesh->triangles[obj_index[i]];
		bboxR.fit(tri.positions[0]);
		bboxR.fit(tri.positions[1]);
		bboxR.fit(tri.positions[2]);
	}

	bestbboxL = bboxL;
	bestbboxR = bboxR;
#else
	// implelement SAH-BVH here
	int* sorted_obj_index[3];
	sorted_obj_index[0] = new int[obj_num];
	sorted_obj_index[1] = new int[obj_num];
	sorted_obj_index[2] = new int[obj_num];
	//std::cout << "numObj: " << obj_num << std::endl;

	float minCost = obj_num; // c_nosplit = # of objects, assuming C_o and C_b are constant time
	int bestIndex = -1;
	int bestSplit = -1;
	int bestAxis = 0;
	AABB bestbboxL, bestbboxR;

	// for each axis
	for (int axis = 0; axis < 3; axis++) {
		AABB* bboxesL = new AABB[2 * obj_num];
		AABB* bboxesR = new AABB[2 * obj_num];


		// construct array of all possible split events
		BVHSplit* splitEvents = new BVHSplit[2 * obj_num];
		for (int i = 0; i < obj_num; i++) {
			const Triangle& t = triangleMesh->triangles[obj_index[i]];
			splitEvents[2 * i].isStart = true;
			splitEvents[2 * i].pos = t.bbox.get_minp()[axis];
			splitEvents[2 * i].obj = &t;
			splitEvents[2 * i].obj_ind = i;
			splitEvents[2 * i + 1].isStart = false;
			splitEvents[2 * i + 1].pos = t.bbox.get_maxp()[axis];
			splitEvents[2 * i + 1].obj = &t;
			splitEvents[2 * i + 1].obj_ind = i;
			splitEvents[2 * i].sorted_ind = new int[1];
			splitEvents[2 * i + 1].sorted_ind = splitEvents[2 * i].sorted_ind;
		}

		std::stable_sort(splitEvents, splitEvents + (obj_num * 2), [](BVHSplit event1, BVHSplit event2)->bool{return event1.pos < event2.pos;});

		// std::cout << "here" << std::endl;
		int sorted_ind = 0;
		for (int i = 0; i < 2 * obj_num; i++) {
			BVHSplit event = splitEvents[i];
			if (event.isStart) {
				// std::cout << event.obj_ind << ", " << event.sorted_ind << std::endl;
				*event.sorted_ind = sorted_ind;
				sorted_obj_index[axis][sorted_ind] = obj_index[event.obj_ind];
				sorted_ind++;
			}
		}
		// std::cout << "here" << std::endl;

		for (int i = 0; i < 2 * obj_num; i++) {
			// std::cout << i << ", " << 2 * obj_num - i - 1 << std::endl;
			BVHSplit &splitL = splitEvents[i];
			BVHSplit &splitR = splitEvents[2 * obj_num - i - 1];

			// std::cout << splitR.pos << ", " << splitR.obj_ind << ", " << splitR.isStart << std::endl;

			// std::cout << "here" << std::endl;
			if (i > 0) {
				bboxesL[i].fit(bboxesL[i - 1].get_minp());
				bboxesL[i].fit(bboxesL[i - 1].get_maxp());
				bboxesL[i].count = bboxesL[i - 1].count;
				bboxesR[2 * obj_num - i - 1].count = bboxesR[2 * obj_num - i].count;
				if (bboxesR[2 * obj_num - i].get_minp() < float3(FLT_MAX)) {
					bboxesR[2 * obj_num - i - 1].fit(bboxesR[2 * obj_num - i].get_minp());
					bboxesR[2 * obj_num - i - 1].fit(bboxesR[2 * obj_num - i].get_maxp());
				}
			}
			// std::cout << "here" << std::endl;

			bboxesL[i].fit(splitL.obj->bbox.get_minp());
			bboxesL[i].fit(splitL.obj->bbox.get_maxp());
			// std::cout << "here" << std::endl;
			if (splitL.isStart) {
				bboxesL[i].count += 1;
			}
			if (splitR.isStart) {
				// std::cout << "fit " << 2 * obj_num - i - 1 << " " << splitR.obj->bbox.get_minp()[axis] << ", " << splitR.obj->bbox.get_maxp()[axis] << std::endl;
				bboxesR[2 * obj_num - i - 1].fit(splitR.obj->bbox.get_minp());
				bboxesR[2 * obj_num - i - 1].fit(splitR.obj->bbox.get_maxp());
				// std::cout << bboxesR[2 * obj_num - i - 1].get_minp()[0] << ", " << bboxesR[2 * obj_num - i - 1].get_minp()[1] << ", " << bboxesR[2 * obj_num - i - 1].get_minp()[2] << std::endl;
				// std::cout << bboxesR[2 * obj_num - i - 1].get_maxp()[0] << ", " << bboxesR[2 * obj_num - i - 1].get_maxp()[1] << ", " << bboxesR[2 * obj_num - i - 1].get_maxp()[2] << std::endl;
				bboxesR[2 * obj_num - i - 1].count += 1;
			}
		}

		for (int split = 0; split < 2 * obj_num - 1; split++) {
			// evaluate the SAH estimation for each split
			// std::cout << bboxesL[split].count << " " << bboxesL[split].area() << ", " << bboxesR[split].count << " " << bboxesR[split].area() << std::endl;
			float c_split = 2 + (bboxesL[split].count * bboxesL[split].area() + bboxesR[split].count * bboxesR[split].area()) / bbox.area();
			// std::cout << "costs " << minCost << " " << c_split << std::endl;
			if (c_split < minCost) {
				// std::cout << "BEST " << axis << ", cost: " << c_split << ", index: " << sorted_ind - 1 << ", split: " << split <<", obj_ind: " << splitEvents[split].obj_ind << ", sorted_ind: " << *splitEvents[split].sorted_ind << std::endl;
				bestAxis = axis;
				minCost = c_split;
				bestIndex = bboxesL[split].count - 1;
				bestSplit = split;
			}
		}
		// std::cout << "here" << std::endl;
		//std::cout << std::endl;

		bestbboxL.fit(bboxesL[bestSplit].get_minp());
		bestbboxL.fit(bboxesL[bestSplit].get_maxp());
		bestbboxR.fit(bboxesR[bestSplit].get_minp());
		bestbboxR.fit(bboxesR[bestSplit].get_maxp());

		for (int i = 0; i < 2 * obj_num; i++) {
			if (splitEvents[i].isStart) {
				delete[] splitEvents[i].sorted_ind;
			}
		}
		delete[] splitEvents;
		delete[] bboxesL;
		delete[] bboxesR;
	}

	// std::cout << "sorted x ";
	// print_arr1(sorted_obj_index[0], obj_num);
	// std::cout << "sorted y ";
	// print_arr1(sorted_obj_index[1], obj_num);
	// std::cout << "sorted z ";
	// print_arr1(sorted_obj_index[2], obj_num);

	// std::cout << "bestIndex " << bestIndex << std::endl;


#endif
	if (obj_num <= 4 || bestIndex == -1 || bestIndex == obj_num - 1) {
		// std::cout << "leaf" << std::endl;

		for (auto & i : sorted_obj_index) {
			delete[] i;
		}

		this->nodeNum++;
		this->node[this->nodeNum - 1].bbox = bbox;
		this->node[this->nodeNum - 1].isLeaf = true;
		this->node[this->nodeNum - 1].triListNum = obj_num;
		this->node[this->nodeNum - 1].triList = new int[obj_num];
		for (int i = 0; i < obj_num; i++) {
			this->node[this->nodeNum - 1].triList[i] = obj_index[i];
		}
		int temp_id;
		temp_id = this->nodeNum - 1;
		this->leafNum++;
		this->levels += level;

		// printf("leaf level: %d \n", level);

		return temp_id;
	} else {
		// std::cout << "bestIndex: " << bestIndex << std::endl;
		// split obj_index into two 
		int* obj_indexL = new int[bestIndex + 1];
		int* obj_indexR = new int[obj_num - (bestIndex + 1)];
		for (int i = 0; i <= bestIndex; ++i) {
			obj_indexL[i] = sorted_obj_index[bestAxis][i];
		}
		for (int i = bestIndex + 1; i < obj_num; ++i) {
			obj_indexR[i - (bestIndex + 1)] = sorted_obj_index[bestAxis][i];
		}

		for (auto & i : sorted_obj_index) {
			delete[] i;
		}

		int obj_numL = bestIndex + 1;
		int obj_numR = obj_num - (bestIndex + 1);

		// recursive call to build a tree
		this->nodeNum++;
		int temp_id;
		temp_id = this->nodeNum - 1;
		this->node[temp_id].bbox = bbox;
		this->node[temp_id].isLeaf = false;
		// std::cout << "numL: " << obj_numL << ", numR: " << obj_numR << std::endl;
		this->node[temp_id].idLeft = splitBVH(obj_indexL, obj_numL, bestbboxL, level + 1);
		this->node[temp_id].idRight = splitBVH(obj_indexR, obj_numR, bestbboxR, level + 1);

		delete[] obj_indexL;
		delete[] obj_indexR;

		return temp_id;
	}
}


int init = 0;
// you may keep this part as-is
void BVH::build(const TriangleMesh* mesh) {
	triangleMesh = mesh;

	// construct the bounding volume hierarchy
	const int obj_num = (int)(triangleMesh->triangles.size());
	int* obj_index = new int[obj_num];
	for (int i = 0; i < obj_num; ++i) {
		obj_index[i] = i;
	}
	this->levels = 0;
	this->nodeNum = 0;
	this->node = new BVHNode[obj_num * 2];
	this->leafNum = 0;

	// calculate a scene bounding box
	AABB bbox;
	for (int i = 0; i < obj_num; i++) {
		const Triangle& tri = triangleMesh->triangles[obj_index[i]];

		bbox.fit(tri.positions[0]);
		bbox.fit(tri.positions[1]);
		bbox.fit(tri.positions[2]);
	}

	// ---------- buliding BVH ----------
	if (!init) {
		printf("Building BVH...\n");
	}
	//std::cout << "SPLIT: " << obj_num << std::endl;
	splitBVH(obj_index, obj_num, bbox);
	if (!init) {
		printf("Done.\n");
	}
	delete[] obj_index;
}


// you may keep this part as-is
bool BVH::traverse(HitInfo& minHit, const Ray& ray, int node_id, float tMin, float tMax, int ignoreParticleId = -1) const {
	bool hit = false;
	HitInfo tempMinHit, tempMinHitL, tempMinHitR;
	bool hit1, hit2;

	if (this->node[node_id].isLeaf) {
		for (int i = 0; i < (this->node[node_id].triListNum); ++i) {
			if ((ignoreParticleId == -1 || triangleMesh->triangles[this->node[node_id].triList[i]].particleId != ignoreParticleId) && triangleMesh->raytraceTriangle(tempMinHit, ray, triangleMesh->triangles[this->node[node_id].triList[i]], tMin, tMax)) {
				hit = true;
				if (tempMinHit.t < minHit.t) minHit = tempMinHit;
			}
		}
	} else {
		hit1 = this->node[this->node[node_id].idLeft].bbox.intersect(tempMinHitL, ray);
		hit2 = this->node[this->node[node_id].idRight].bbox.intersect(tempMinHitR, ray);

		hit1 = hit1 && (tempMinHitL.t < minHit.t);
		hit2 = hit2 && (tempMinHitR.t < minHit.t);

		if (hit1 && hit2) {
			if (tempMinHitL.t < tempMinHitR.t) {
				hit = traverse(minHit, ray, this->node[node_id].idLeft, tMin, tMax, ignoreParticleId);
				hit = traverse(minHit, ray, this->node[node_id].idRight, tMin, tMax, ignoreParticleId);
			} else {
				hit = traverse(minHit, ray, this->node[node_id].idRight, tMin, tMax, ignoreParticleId);
				hit = traverse(minHit, ray, this->node[node_id].idLeft, tMin, tMax, ignoreParticleId);
			}
		} else if (hit1) {
			hit = traverse(minHit, ray, this->node[node_id].idLeft, tMin, tMax, ignoreParticleId);
		} else if (hit2) {
			hit = traverse(minHit, ray, this->node[node_id].idRight, tMin, tMax, ignoreParticleId);
		}
	}

	return hit;
}




class AreaLightSource {
public:
	float3 intensity = float3(100.0f);
	float3 vertices[4];
	TriangleMesh* mesh;
	bool isActive = true;
};


// scene definition
class Scene {
public:
	std::vector<TriangleMesh*> objects;
	std::vector<PointLightSource*> pointLightSources;
	std::vector<AreaLightSource*> areaLightSources;
	std::vector<BVH> bvhs;
	std::vector<BVH> areaLightbvhs;
	Image envMap;
	std::chrono::high_resolution_clock clock;

	void addObject(TriangleMesh* pObj) {
		objects.push_back(pObj);
	}
	void addLight(PointLightSource* pObj) {
		pointLightSources.push_back(pObj);
	}

	void addLight(AreaLightSource* arealight) {
		areaLightSources.push_back(arealight);
	}

	void preCalc() {
		bvhs.resize(objects.size());
		areaLightbvhs.resize(areaLightSources.size());

		std::chrono::time_point start = clock.now();

		for (int i = 0; i < objects.size(); i++) {
			objects[i]->preCalc();
			bvhs[i].build(objects[i]);
			if (!init) {
				printf("bvh num leafs: %d\n", bvhs[i].leafNum);
				printf("bvh avg depth: %d\n", bvhs[i].levels / bvhs[i].leafNum);
			}
		}

		for (int i = 0; i < areaLightSources.size(); i++) {
			areaLightSources[i]->mesh->preCalc();
			areaLightbvhs[i].build(areaLightSources[i]->mesh);
			if (!init) {
				printf("area light bvh num leafs: %d\n", areaLightbvhs[i].leafNum);
				printf("area light bvh avg depth: %d\n", areaLightbvhs[i].levels / areaLightbvhs[i].leafNum);
			}
		}

		std::chrono::time_point end = clock.now();
		if (!init) {
			std::cout << "building bvh took (ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
		}
		init = 1;
	}

	// ray-scene intersection
	bool intersect(HitInfo& minHit, const Ray& ray, float tMin = 0.0f, float tMax = FLT_MAX, bool isShadowTrace = false, int ignoreParticleId = -1) const {
		bool hit = false;
		HitInfo tempMinHit;
		minHit.t = FLT_MAX;

		if (!isShadowTrace) {
			for (int i = 0, i_n = (int)areaLightSources.size(); i < i_n; i++) {
				if (areaLightbvhs[i].intersect(tempMinHit, ray, tMin, tMax, ignoreParticleId)) {
					if (tempMinHit.t < minHit.t) {
						hit = true;
						tempMinHit.isLight = true;
						tempMinHit.intensity = areaLightSources[i]->intensity;
						minHit = tempMinHit;
					}
				}
			}
		}

		for (int i = 0, i_n = (int)objects.size(); i < i_n; i++) {
			//if (objects[i]->bruteforceIntersect(tempMinHit, ray, tMin, tMax)) { // for debugging
			if (bvhs[i].intersect(tempMinHit, ray, tMin, tMax, ignoreParticleId)) {
				if (tempMinHit.t < minHit.t) {
					hit = true;
					minHit = tempMinHit;
				}
			}
		}
		return hit;
	}

	// camera -> screen matrix (given to you for A1)
	float4x4 perspectiveMatrix(float fovy, float aspect, float zNear, float zFar) const {
		float4x4 m;
		const float f = 1.0f / (tan(fovy * DegToRad / 2.0f));
		m[0] = { f / aspect, 0.0f, 0.0f, 0.0f };
		m[1] = { 0.0f, f, 0.0f, 0.0f };
		m[2] = { 0.0f, 0.0f, (zFar + zNear) / (zNear - zFar), -1.0f };
		m[3] = { 0.0f, 0.0f, (2.0f * zFar * zNear) / (zNear - zFar), 0.0f };

		return m;
	}

	// model -> camera matrix (given to you for A1)
	float4x4 lookatMatrix(const float3& _eye, const float3& _center, const float3& _up) const {
		// transformation to the camera coordinate
		float4x4 m;
		const float3 f = normalize(_center - _eye);
		const float3 upp = normalize(_up);
		const float3 s = normalize(cross(f, upp));
		const float3 u = cross(s, f);

		m[0] = { s.x, s.y, s.z, 0.0f };
		m[1] = { u.x, u.y, u.z, 0.0f };
		m[2] = { -f.x, -f.y, -f.z, 0.0f };
		m[3] = { 0.0f, 0.0f, 0.0f, 1.0f };
		m = transpose(m);

		// translation according to the camera location
		const float4x4 t = float4x4{ {1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 0.0f}, { -_eye.x, -_eye.y, -_eye.z, 1.0f} };

		m = mul(m, t);
		return m;
	}

	// rasterizer
	void Rasterize() const {
		// ====== implement it in A1 ======
		// fill in plm by a proper matrix
		const float4x4 pm = perspectiveMatrix(globalFOV, globalAspectRatio, globalDepthMin, globalDepthMax);
		const float4x4 lm = lookatMatrix(globalEye, globalLookat, globalUp);
		const float4x4 plm = mul(pm, lm);

		FrameBuffer.clear();
		for (int n = 0, n_n = (int)objects.size(); n < n_n; n++) {
			for (int k = 0, k_n = (int)objects[n]->triangles.size(); k < k_n; k++) {
				objects[n]->rasterizeTriangle(objects[n]->triangles[k], plm);
			}
		}
	}

	// eye ray generation (given to you for A2)
	Ray eyeRay(int x, int y, bool differentials = false) const {
		// compute the camera coordinate system 
		const float3 wDir = normalize(float3(-globalViewDir));
		const float3 uDir = normalize(cross(globalUp, wDir));
		const float3 vDir = cross(wDir, uDir);

		// compute the pixel location in the world coordinate system using the camera coordinate system
		// trace a ray through the center of each pixel
		const float imPlaneUPos = (x + 0.5f) / float(globalWidth) - 0.5f;
		const float imPlaneVPos = (y + 0.5f) / float(globalHeight) - 0.5f;

		const float3 pixelPos = globalEye + float(globalAspectRatio * globalFilmSize * imPlaneUPos) * uDir + float(globalFilmSize * imPlaneVPos) * vDir - globalDistanceToFilm * wDir;

		Ray r = Ray(globalEye, normalize(pixelPos - globalEye));
		if (differentials) {
			generateEyeRayDifferentials(x, y, r);
		}
		return r;
	}

	void generateEyeRayDifferentials(int x, int y, Ray& mainRay) const {
		// compute the camera coordinate system
		const float3 wDir = normalize(float3(-globalViewDir));
		const float3 uDir = normalize(cross(globalUp, wDir));
		const float3 vDir = cross(wDir, uDir);
		float3 diffRight = normalize(cross(-globalViewDir, globalUp));

		Ray rx = eyeRay(x + 1 > globalWidth ? x - 1 : x + 1, y);
		Ray ry = eyeRay(x, y + 1 > globalHeight ? y - 1 : y + 1);

		// set differentials
		mainRay.rxOrigin = rx.o;
		mainRay.ryOrigin = ry.o;
		mainRay.rxDir = rx.d;
		mainRay.ryDir = ry.d;
		mainRay.dpdx = float3(0.0f);
		mainRay.dpdy = float3(0.0f);
		mainRay.dddx = (dot(rx.d, rx.d) * diffRight - dot(rx.d, diffRight) * rx.d) / pow(dot(rx.d, rx.d), 1.5f);
		mainRay.dddy = (dot(ry.d, ry.d) * globalUp - dot(ry.d, globalUp) * ry.d) / pow(dot(ry.d, ry.d), 1.5f);
		mainRay.hasDifferentials = true;
	}

	float3 environmentMap(const float3& viewDir) {
		float r = (1 / PI) * acos(viewDir.z) / sqrt(pow(viewDir.x, 2) + pow(viewDir.y, 2));
		int x = int(viewDir.x * r * envMap.width / 2 + envMap.width / 2);
		int y = int(viewDir.y * r * envMap.height / 2 + envMap.height / 2);
		//std::cout << x * r << ", " << y * r << ", " << envMap.valid(x, y) << std::endl;
		if (envMap.valid(x, y)) {
			return envMap.pixel(x, y);
		}
		return float3(0.0f);
	}

	// ray tracing (you probably don't need to change it in A2)
	void Raytrace() const {
		FrameBuffer.clear();

		std::chrono::time_point start = clock.now();
		// loop over all pixels in the image
		for (int j = 0; j < globalHeight; ++j) {
			for (int i = 0; i < globalWidth; ++i) {
				const Ray ray = eyeRay(i, j, true);
				HitInfo hitInfo;
				//std::cout << i << ", " << j << std::endl;
				float3 diffRight = normalize(cross(-globalViewDir, globalUp));
				//hitInfo.rxOrigin = ray.rxOrigin;
				//hitInfo.ryOrigin = ray.ryOrigin;
				//hitInfo.rxDir = ray.rxDir;
				//hitInfo.ryDir = ray.ryDir;
				//hitInfo.dpdx = float3(0.0f);
				//hitInfo.dpdy = float3(0.0f);
				//hitInfo.dddx = (dot(ray.rxDir, ray.rxDir) * diffRight - dot(ray.rxDir, diffRight) * ray.rxDir) / pow(dot(ray.rxDir, ray.rxDir), 1.5f);
				//hitInfo.dddy = (dot(ray.ryDir, ray.ryDir) * globalUp - dot(ray.ryDir, globalUp) * ray.ryDir) / pow(dot(ray.ryDir, ray.ryDir), 1.5f);
				//std::cout << "diff rays " << hitInfo.rxOrigin[0] << ", " << hitInfo.rxOrigin[1] << " " << hitInfo.rxDir[0] << ", " << hitInfo.rxDir[1] << std::endl;
				if (intersect(hitInfo, ray)) {
					FrameBuffer.pixel(i, j) = shade(hitInfo, -ray.d);
				}
				else {
					FrameBuffer.pixel(i, j) = shade(hitInfo, -ray.d, -1);
				}
			}

			// show intermediate process
			if (globalShowRaytraceProgress) {
				constexpr int scanlineNum = 64;
				if ((j % scanlineNum) == (scanlineNum - 1)) {
					glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, globalWidth, globalHeight, GL_RGB, GL_FLOAT, &FrameBuffer.pixels[0]);
					glRecti(1, 1, -1, -1);
					glfwSwapBuffers(globalGLFWindow);
					printf("Rendering Progress: %.3f%%\r", j / float(globalHeight - 1) * 100.0f);
					fflush(stdout);
				}
			}
		}

		std::chrono::time_point end = clock.now();

		std::cout << "msec/frame: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
	}

};
static Scene globalScene;



int a3TaskRunMode = 2;
float CR = 0.4;
// ====== implement it in A3 ======
// fill in the missing parts
class Particle {
public:
	float3 position = float3(0.0f);
	float3 velocity = float3(0.0f);
	float3 prevPosition = position;
	float3 accumulatedForce = float3(0.0f);
	float radius;
	float mass = 3;
	int particleId;

	void reset() {
		position = float3(PCG32::rand() + (testCollisions ? 0.0f : 1.5f), PCG32::rand() + 0.5f, PCG32::rand()) - float(0.5f);
		velocity = 2.0f * float3((PCG32::rand() - 0.5f), (PCG32::rand() - 0.5f), (PCG32::rand() - 0.5f));
		prevPosition = position;
		position += velocity * deltaT;
	}

	void step() {
		float3 temp = position;
		bool collision = false;
		//position += velocity * deltaT;
		position += testCollisions ? (temp - prevPosition) + pow(deltaT, 2.0f) * globalGravity * mass : velocity * deltaT;
		prevPosition = temp;

		float3 posVec = position - prevPosition;
		float3 newOrigin = prevPosition + (radius)*normalize(posVec);
		Ray velRay = Ray(prevPosition, normalize(posVec));
		HitInfo collisionHit;
		if (!NO_COLLIDE_FLAG && globalScene.intersect(collisionHit, velRay, 0, length(posVec) + radius, false, particleId)) {
			//printf("%f %f %f | %f %f %f\n", prevPosition[0], prevPosition[1], prevPosition[2], newOrigin[0], newOrigin[1], newOrigin[2]);
			//printf("%f\n", collisionHit.t);
			float3 facingNormal = dot(collisionHit.N, -posVec) < 0 ? -collisionHit.N : collisionHit.N;
			collision = true;
			float3 newDir = normalize(-2 * (dot(normalize(posVec), facingNormal)) * facingNormal + normalize(posVec));
			position = collisionHit.P + collisionHit.N * (radius);
			prevPosition = position - length(posVec) * newDir * CR;
		}
		velocity = (position - prevPosition) / deltaT;
	}
};


class ParticleSystem {
public:
	std::vector<Particle> particles;
	TriangleMesh particlesMesh;
	TriangleMesh sphere;
	TriangleMesh shape;
	const char* sphereMeshFilePath = 0;
	float sphereSize = 0.0f;
	float shapeDistances[6];
	ParticleSystem() {};

	void updateMesh() {
		// you can optionally update the other mesh information (e.g., bounding box, BVH - which is tricky)
		if (sphereSize > 0) {
			const int n = int(sphere.triangles.size());
			for (int i = 0; i < globalNumParticles; i++) {
				for (int j = 0; j < n; j++) {
					particlesMesh.triangles[i * n + j].positions[0] = sphere.triangles[j].positions[0] + particles[i].position;
					particlesMesh.triangles[i * n + j].positions[1] = sphere.triangles[j].positions[1] + particles[i].position;
					particlesMesh.triangles[i * n + j].positions[2] = sphere.triangles[j].positions[2] + particles[i].position;
					particlesMesh.triangles[i * n + j].normals[0] = sphere.triangles[j].normals[0];
					particlesMesh.triangles[i * n + j].normals[1] = sphere.triangles[j].normals[1];
					particlesMesh.triangles[i * n + j].normals[2] = sphere.triangles[j].normals[2];
					if (!init) {
						particlesMesh.triangles[i * n + j].particleId = particles[i].particleId;
					}
				}
			}
		} else {
			const float particleSize = 0.005f;
			for (int i = 0; i < globalNumParticles; i++) {
				// facing toward the camera
				particlesMesh.triangles[i].positions[0] = particles[i].position;
				particlesMesh.triangles[i].positions[1] = particles[i].position + particleSize * globalUp;
				particlesMesh.triangles[i].positions[2] = particles[i].position + particleSize * globalRight;
				particlesMesh.triangles[i].normals[0] = -globalViewDir;
				particlesMesh.triangles[i].normals[1] = -globalViewDir;
				particlesMesh.triangles[i].normals[2] = -globalViewDir;
			}
		}
		globalScene.preCalc();
	}

	void initialize() {
		particles.resize(globalNumParticles);
		particlesMesh.materials.resize(1);
		for (int i = 0; i < globalNumParticles; i++) {
			particles[i].reset();
		}

		float3 shapeVertices1[4];
		float3 shapeVertices2[4];
		shapeVertices1[0] = float3(1.0, 0, -0.5);
		shapeVertices1[1] = float3(1.0, 0, 0.5);
		shapeVertices1[2] = float3(2.0, 0, 0.5);
		shapeVertices1[3] = float3(2.0, 0, -0.5);
		shapeVertices2[0] = float3(1.0, 1, -0.5);
		shapeVertices2[1] = float3(1.0, 1, 0.5);
		shapeVertices2[2] = float3(2.0, 1, 0.5);
		shapeVertices2[3] = float3(2.0, 1, -0.5);
		shape.createSingleQuad(shapeVertices1);
		shape.createSingleQuad(shapeVertices2);



		shapeDistances[0] = length(particles[0].position - particles[1].position);
		shapeDistances[1] = length(particles[0].position - particles[2].position);
		shapeDistances[2] = length(particles[0].position - particles[3].position);
		shapeDistances[3] = length(particles[1].position - particles[2].position);
		shapeDistances[4] = length(particles[1].position - particles[3].position);
		shapeDistances[5] = length(particles[2].position - particles[3].position);

		if (sphereMeshFilePath) {
			if (sphere.load(sphereMeshFilePath) || sphere.load("../media/smallsphere.obj")) {
				particlesMesh.triangles.resize(sphere.triangles.size() * globalNumParticles);
				sphere.preCalc();
				sphereSize = sphere.bbox.get_size().x * 0.5f;
				for (int i = 0; i < globalNumParticles; i++) {
					particles[i].radius = sphereSize;
					particles[i].particleId = i;
				}
			} else {
				particlesMesh.triangles.resize(globalNumParticles);
			}
		} else {
			particlesMesh.triangles.resize(globalNumParticles);
		}
		updateMesh();
	}

	void step() {
		float4x4 RTheta = {
			{cos(0.01f), 0, -sin(0.01f), 0},
			{0, 1, 0, 0},
			{sin(0.01f), 0, cos(0.01f), 0},
			{0, 0, 0, 1}
		};
		float3 center = shape.getCenter();
		float4x4 T1 = {
			{1, 0, 0, 0},
			{0, 1, 0, 0},
			{0, 0, 1, 0},
			{-center[0], -center[1], -center[2], 1}
		};
		float4x4 T1Inv = {
			{1, 0, 0, 0},
			{0, 1, 0, 0},
			{0, 0, 1, 0},
			{center[0], center[1], center[2], 1}
		};
		shape.transform(T1);
		shape.transform(RTheta);
		shape.transform(T1Inv);
		if (globalParticleConstraint) {
			float3 beforePos[4];
			beforePos[0] = particles[0].position;
			beforePos[1] = particles[1].position;
			beforePos[2] = particles[2].position;
			beforePos[3] = particles[3].position;
			float3 cmass = (beforePos[0] + beforePos[1] + beforePos[2] + beforePos[3]) / 4;
			for (int i = 0; i < 4; i++) {
				particles[i].step();
			}
			for (int it = 0; it < 50; it++) {
				// Sequential rigid body distance constraints
				for (int i = 0; i < 3; i++) {
					for (int j = i + 1; j < 4; j++) {
						int distInd = i == 0 ? j - 1 : i + j;
						if (length(particles[i].position - particles[j].position) != shapeDistances[distInd]) {
							float3 dp = particles[j].position - particles[i].position;
							float3 deltaX = dp - normalize(dp) * shapeDistances[distInd];
							float weight = particles[i].mass / (particles[i].mass + particles[j].mass);
							particles[i].position = particles[i].position + deltaX * weight;
							particles[j].position = particles[j].position - deltaX * (1.0f - weight);
						}
					}
				}
			}
			float3 t = (particles[0].position + particles[1].position + particles[2].position + particles[3].position) / 4;

			// Below is attempted shape matching by finding optimal rotation matrix
			float3x3 Apq;
			float3x4 pi;
			float3x4 qi;
			for (int i = 0; i < 4; i++) {
				pi[i] = particles[i].position - t;
				qi[i] = beforePos[i] - cmass;
				float3x3 pq = {
					{pi[i][0] * qi[i][0], pi[i][1] * qi[i][0], pi[i][2] * qi[i][0]},
					{pi[i][0] * qi[i][1], pi[i][1] * qi[i][1], pi[i][2] * qi[i][1]},
					{pi[i][0] * qi[i][2], pi[i][1] * qi[i][2], pi[i][2] * qi[i][2]},
				};
				Apq += particles[i].mass * pq;// mul(transpose(transpose(pi[i])), transpose(qi[i]));
			}
			//Apq = mul(pi, transpose(qi));
			float3x3 S = sqrt(mul(transpose(Apq), Apq));
			float3x3 R = mul(Apq, inverse(S));

			float3 tt = t - cmass;

			//printf("apq:\n%f %f %f\n%f %f %f\n%f %f %f\n", Apq[0][0], Apq[0][1], Apq[0][2], Apq[1][0], Apq[1][1], Apq[1][2], Apq[2][0], Apq[2][1], Apq[2][2]);
			//printf("r:\n%f %f %f\n%f %f %f\n%f %f %f\n", R[0][0], R[0][1], R[0][2], R[1][0], R[1][1], R[1][2], R[2][0], R[2][1], R[2][2]);
			float4x4 R4 = {
				{R[0][0], R[0][1], R[0][2], 0},
				{R[1][0], R[1][1], R[2][2], 0},
				{R[2][0], R[2][1], R[1][2], 0},
				{0, 0, 0, 1}
			};
			float4x4 T2 = {
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{0, 0, 1, 0},
				{tt[0], tt[1], tt[2], 1}
			};
			for (int i = 4; i < globalNumParticles; i++) {
				particles[i].position = particles[i].position - cmass + t;
			}
			shape.transform(T2);
		} else {
			for (int i = 0; i < globalNumParticles; i++) {
				particles[i].step();
			}
		}
		updateMesh();
	}
};
static ParticleSystem globalParticleSystem;


void generateReflectedDifferentials(const HitInfo& hit, Ray& reflectedRay, float3 N, float3 wi, float dDNdx, float dDNdy) {
	reflectedRay.dpdx = hit.dpdx;
	reflectedRay.dpdy = hit.dpdy;
	reflectedRay.dddx = hit.dddx - 2 * (dot(wi, N) * hit.dNdx + dDNdx * N);
	reflectedRay.dddy = hit.dddy - 2 * (dot(wi, N) * hit.dNdy + dDNdy * N);
	reflectedRay.rxOrigin = hit.P + hit.dpdx;
	reflectedRay.ryOrigin = hit.P + hit.dpdy;
	reflectedRay.rxDir = reflectedRay.d + reflectedRay.dddx;
	reflectedRay.ryDir = reflectedRay.d + reflectedRay.dddy;
	reflectedRay.hasDifferentials = true;
}

void generateRefractedDifferentials(const HitInfo& hit, Ray& refractedRay, float3 N, float3 wi, float eta, float DprimeN, float dDNdx, float dDNdy) {
	float mu = eta * dot(wi, N) - DprimeN;
	float dmudx = (eta - (pow(eta, 2.0f) * dot(wi, N)) / DprimeN) * dDNdx;
	float dmudy = (eta - (pow(eta, 2.0f) * dot(wi, N)) / DprimeN) * dDNdy;
	refractedRay.dpdx = hit.dpdx;
	refractedRay.dpdy = hit.dpdy;
	refractedRay.dddx = eta * hit.dddx - (mu * hit.dNdx + dmudx * N);
	refractedRay.dddy = eta * hit.dddy - (mu * hit.dNdy + dmudy * N);
	refractedRay.rxOrigin = hit.P + hit.dpdx;
	refractedRay.ryOrigin = hit.P + hit.dpdy;
	refractedRay.rxDir = refractedRay.d + refractedRay.dddx;
	refractedRay.ryDir = refractedRay.d + refractedRay.dddy;
	refractedRay.hasDifferentials = true;
}

// ====== implement it in A2 ======
// fill in the missing parts
static float3 shade(const HitInfo& hit, const float3& viewDir, const int level) {
	assert(length(hit.d) == 1);
	assert(length(viewDir) == 1);
	if (hit.isLight) {
		return hit.intensity;
	}
	if (level == -1) {
		if (!enableEnvironmentMapping) {
			return float3(0.4f);
		} else {
			return globalScene.environmentMap(-viewDir);
		}
	}
	if (level > 5) {
		return float3(0.0f);
	}
	if (hit.material->type == MAT_LAMBERTIAN) {
		// you may want to add shadow ray tracing here in A2
		float3 L = float3(0.0f);
		float3 brdf, irradiance = float3(0.0f), tex = float3(1.0f);

		if (hit.material->isTextured) {
			// std::cout << hit.dudx << " " << hit.dvdx << " " << hit.dudy << " " << hit.dvdy << std::endl;
			float lod = std::min(float(hit.material->numMaps - 1), std::max(0.0f, log2(std::max(sqrt(std::pow(hit.dudx, 2.0f) + std::pow(hit.dvdx, 2.0f)), sqrt(std::pow(hit.dudy, 2.0f) + std::pow(hit.dvdy, 2.0f))))));
			/*if (lod > 0) {
				std::cout << "lod" << lod << std::endl;
			}*/

			//brdf *= hit.material->fetchTexture(hit.T);
			if (globalUseMipMap) {
				tex = hit.material->fetchTexture(hit.T, lod);
			} else {
				tex = hit.material->fetchTexture(hit.T);
			}
			// float dTdx = sqrt(std::pow(hit.dudx, 2.0f) + std::pow(hit.dvdx, 2.0f));
			// float dTdy = sqrt(std::pow(hit.dudy, 2.0f) + std::pow(hit.dvdy, 2.0f));
			// float lod;
			// float du, dv;
			// if (dTdx > dTdy) {
			// 	lod = log2(dTdy);
			// 	du = hit.dudx;
			// 	dv = hit.dvdx;
			// } else {
			// 	lod = log2(dTdx);
			// 	du = hit.dudy;
			// 	dv = hit.dvdy;
			// }
			// lod = std::min(float(hit.material->numMaps - 1), std::max(0.0f, lod));
			// int numAnisotropicSamples = 5;
			// float2 texel = hit.T;
			// float3 avgSample = float3(0.0f);
			// for (int j = 0; j < numAnisotropicSamples; j++) {
			// 	float t = float(j) / float(numAnisotropicSamples - 1);
			// 	texel.x += t * du;
			// 	texel.y += t * dv;
			// 	avgSample += hit.material->fetchTexture(texel, lod);
			// }
			// avgSample /= numAnisotropicSamples;
			// brdf *= avgSample;
		}

		// loop over all of the point light sources
		int numPointLights = globalScene.pointLightSources.size();
		int numAreaLights = globalScene.areaLightSources.size();
		float3 facingNormal = hit.N;
		if (dot(hit.N, viewDir) < 0) {
			facingNormal = -hit.N;
		}
		for (int i = 0; i < numPointLights + numAreaLights; i++) {
			bool isAreaLight = false;
			if (i >= numPointLights) {
				isAreaLight = true;
			}
			//return tex * hit.material->BRDF(globalScene.pointLightSources[i]->position - hit.P, viewDir, hit.N) * PI; //debug output
			std::vector<bool> triangleVisible;
			bool lightVisible = false;
			if (isAreaLight) {
				if (!globalScene.areaLightSources[i - numPointLights]->isActive) {
					continue;
				}
				// do a visibility test for all triangles in the area light
				for (int triInd = 0; triInd < globalScene.areaLightSources[i - numPointLights]->mesh->triangles.size(); triInd++) {
					for (int v = 0; v < 3; v++) {
						float3 l = globalScene.areaLightSources[i - numPointLights]->mesh->triangles[triInd].positions[v] - hit.P;
						Ray shadowRay = Ray(hit.P + Epsilon * facingNormal, normalize(l));

						HitInfo shadowHit;
						if (!globalScene.intersect(shadowHit, shadowRay, 0, length(l), true)) {
							triangleVisible.push_back(true);
							lightVisible = true;
							goto nextTriangle;
						}
					}
					triangleVisible.push_back(false);
				nextTriangle:;
				}
				if (!lightVisible) {
					continue;
				}
				if (globalAreaLightMethod == AREA_LIGHT_LTC) {
					float3 ltcNormal = facingNormal;
					const float theta = acos(dot(ltcNormal, viewDir));
					float3x3 Minv = Minv_GGX(theta, globalRoughness, antiAliasMinv);// hit.material->roughness);

					float3 points[4];
					points[0] = globalScene.areaLightSources[i - numPointLights]->vertices[0];
					points[1] = globalScene.areaLightSources[i - numPointLights]->vertices[1];
					points[2] = globalScene.areaLightSources[i - numPointLights]->vertices[2];
					points[3] = globalScene.areaLightSources[i - numPointLights]->vertices[3];
					float3 spec = ltcEvaluate(ltcNormal, viewDir, hit.P, Minv, points, true);
					// printf("%f\n", spec);
					spec *= amplitude_GGX(theta, globalRoughness);//  hit.material->roughness);

					float3x3 id = { {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f} };
					float3 diff = ltcEvaluate(ltcNormal, viewDir, hit.P, id, points, true);

					//printf("%f %f %f\n", diff[0], diff[1], diff[2] );
					float3 col = globalScene.areaLightSources[i - numPointLights]->intensity * (spec + hit.material->Kd * diff) / (2.0f * PI);
					L += col * tex;
					if (col != float3(0.0f)) {
						continue;
					}
					//printf("%f %f %f\n", ltcNormal[0], ltcNormal[1], ltcNormal[2]);
					continue;
				}
			}

			int numAreaLightSamples = 100;
			for (int n = 0; n < numAreaLightSamples; n++) {
				float3 l;
				float3 yN;
				if (isAreaLight) {
					float3 randPoint;
					int randInd;
					std::tie(randPoint, yN, randInd) = globalScene.areaLightSources[i - numPointLights]->mesh->samplePoint();
					if (!triangleVisible[randInd]) {
						continue;
					}
					l = randPoint - hit.P;
				}
				else {
					l = globalScene.pointLightSources[i]->position - hit.P;
				}
				float ldist = length(l);
				// the inverse-squared falloff
				const float falloff = length2(l);

				// normalize the light direction
				l /= sqrtf(falloff);

				Ray shadowRay = Ray(hit.P + Epsilon * facingNormal, normalize(l));

				HitInfo shadowHit;
				if (globalScene.intersect(shadowHit, shadowRay, 0, ldist, true)) {
					if (isAreaLight) {
						continue;
					} else {
						goto skipLight;
					}
				}

				//brdf = hit.material->BRDF(l, viewDir, facingNormal);
				brdf = hit.material->microFacetBRDF(l, viewDir, hit.N);

				// get the irradiance
				if (isAreaLight) {
					float G = std::abs(dot(hit.N, l) * dot(yN, l)) / std::pow(length(l), 2.0f);
					irradiance += brdf / (4.0f * PI * falloff) * float3(1000.0f) * G;
				}
				else {
					irradiance = float(std::max(0.0f, dot(facingNormal, l)) / (4.0 * PI * falloff)) * globalScene.pointLightSources[i]->wattage;
					break;
				}
				//return tex * hit.material->BRDF(l, viewDir, hit.N) * PI; //debug output
			}

			brdf *= tex;

			if (isAreaLight) {
				L += globalScene.areaLightSources[i - numPointLights]->mesh->area * tex * irradiance / numAreaLightSamples;
			} else {
				L += irradiance * brdf;
			}


		skipLight:;
		}
		return L;
	} else if (hit.material->type == MAT_METAL) {
		float3 facingNormal = hit.N;
		float3 geoNormal = hit.geoN;

		// check front/back face
		if (dot(viewDir, hit.N) < 0) {
			facingNormal = -facingNormal;
			geoNormal = -geoNormal;
		}
		Ray reflectedRay = Ray(hit.P + Epsilon * facingNormal, normalize(-2 * (dot(-viewDir, facingNormal)) * facingNormal - viewDir));
		float dDNdx = dot(hit.dddx, facingNormal) + dot(-viewDir, hit.dNdx);
		float dDNdy = dot(hit.dddy, facingNormal) + dot(-viewDir, hit.dNdy);
		generateReflectedDifferentials(hit, reflectedRay, facingNormal, -viewDir, dDNdx, dDNdy);

		// check erroneous reflection from shading normal
		if (dot(reflectedRay.d, geoNormal) < 0) { // if reflectedRay goes into object
			reflectedRay.d = normalize(reflectedRay.d - dot(2 * reflectedRay.d, geoNormal) * geoNormal);
			reflectedRay.hasDifferentials = false;
		}

		HitInfo tempHit;
		if (globalScene.intersect(tempHit, reflectedRay)) {
			// return hit.material->Ks * hit.material->microFacetBRDF(reflectedRay.d, viewDir, hit.N) * shade(tempHit, -reflectedRay.d, level + 1);
			return hit.material->Ks * shade(tempHit, -reflectedRay.d, level + 1);
		}
		if (!enableEnvironmentMapping) {
			return float3(0.0f);
		} else {
			return globalScene.environmentMap(reflectedRay.d);
		}
	} else if (hit.material->type == MAT_GLASS) {
		float3 geoNormal = hit.geoN;
		float3 facingNormal = hit.N;

		// check front/back face
		float etaR = 1 / hit.material->eta;
		float eta2 = hit.material->eta;
		if (dot(viewDir, hit.N) < 0) {
			facingNormal = -facingNormal;
			geoNormal = -geoNormal;
			etaR = hit.material->eta;
			eta2 = 1;
		}
		float innerTerm = 1 - pow(etaR, 2.0f) * (1 - pow(dot(viewDir, facingNormal), 2.0f));

		// calculate reflected ray for both total internal reflection and fresnel
		Ray reflectedRay = Ray(hit.P + Epsilon * facingNormal, normalize(-2 * (dot(-viewDir, facingNormal)) * facingNormal - viewDir));
		float dDNdx = dot(hit.dddx, facingNormal) + dot(-viewDir, hit.dNdx);
		float dDNdy = dot(hit.dddy, facingNormal) + dot(-viewDir, hit.dNdy);
		generateReflectedDifferentials(hit, reflectedRay, facingNormal, -viewDir, dDNdx, dDNdy);

		// check erroneous reflection from shading normal
		if (dot(reflectedRay.d, geoNormal) < 0) { // if reflectedRay crosses surface boundary
			reflectedRay.d = normalize(reflectedRay.d - dot(2 * reflectedRay.d, geoNormal) * geoNormal);
			reflectedRay.hasDifferentials = false;
		}

		if (innerTerm < 0) {
			// total internal reflection
			HitInfo tempHit;
			if (globalScene.intersect(tempHit, reflectedRay)) {
				return shade(tempHit, -reflectedRay.d, level + 1);
			}
			if (!enableEnvironmentMapping) {
				return float3(0.0f);
			} else {
				return globalScene.environmentMap(reflectedRay.d);
			}
			
		} else {
			Ray refractedRay = Ray(hit.P + Epsilon * -facingNormal, normalize(etaR * (-viewDir - (dot(-viewDir, facingNormal) * facingNormal)) - sqrt(innerTerm) * facingNormal));
			generateRefractedDifferentials(hit, refractedRay, facingNormal, -viewDir, etaR, -sqrt(innerTerm), dDNdx, dDNdy);
			// check erroneous refraction from shading normal
			if (dot(refractedRay.d, geoNormal) > 0) { // if refractedRay goes back toward the origin of incoming ray
				// TODO: double check if this is correct for refraction
				refractedRay.d = normalize(refractedRay.d - dot(2 * refractedRay.d, geoNormal) * geoNormal);
				refractedRay.hasDifferentials = false;
			}

			// Fresnel values
			// TODO: double check if this needs to be geometric normal if corrected
			float cosThetai = dot(viewDir, facingNormal);
			float cosThetao = dot(refractedRay.d, -facingNormal);
			float ps = (etaR * eta2 * cosThetai - eta2 * cosThetao) / (etaR * eta2 * cosThetai + eta2 * cosThetao);
			float pt = (etaR * eta2 * cosThetao - eta2 * cosThetai) / (etaR * eta2 * cosThetao + eta2 * cosThetai);
			float fresnel = 0.5f * (ps * ps + pt * pt);
			assert(0.0f <= fresnel && fresnel <= 1.0f);

			// std::cout << cosThetai << ", " << cosThetao << ", " << ps << ", " << pt << ", " << fresnel << ", " << etaR * eta2 << ", " << eta2 << std::endl;
			HitInfo tempHit;
			if (globalScene.intersect(tempHit, refractedRay)) {
				if (globalUseFresnel) {
					return (1.0f - fresnel) * shade(tempHit, -refractedRay.d, level + 1)
					+ fresnel * shade(tempHit, -reflectedRay.d, level + 1);
				} else {
					return shade(tempHit, -refractedRay.d, level + 1);
				}
			}
			if (!enableEnvironmentMapping) {
				return float3(0.0f);
			} else {
				return globalScene.environmentMap(refractedRay.d);
			}
		}
	} else {
		// something went wrong - make it apparent that it is an error
		return float3(100.0f, 0.0f, 100.0f);
	}
}







// OpenGL initialization (you will not use any OpenGL/Vulkan/DirectX... APIs to render 3D objects!)
// you probably do not need to modify this in A0 to A3.
class OpenGLInit {
public:
	OpenGLInit() {
		// initialize GLFW
		if (!glfwInit()) {
			std::cerr << "Failed to initialize GLFW." << std::endl;
			exit(-1);
		}

		// create a window
		glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
		globalGLFWindow = glfwCreateWindow(globalWidth, globalHeight, "Welcome to CS488/688!", NULL, NULL);
		if (globalGLFWindow == NULL) {
			std::cerr << "Failed to open GLFW window." << std::endl;
			glfwTerminate();
			exit(-1);
		}

		// make OpenGL context for the window
		glfwMakeContextCurrent(globalGLFWindow);

		// initialize GLEW
		glewExperimental = true;
		if (glewInit() != GLEW_OK) {
			std::cerr << "Failed to initialize GLEW." << std::endl;
			glfwTerminate();
			exit(-1);
		}

		// set callback functions for events
		glfwSetKeyCallback(globalGLFWindow, keyFunc);
		glfwSetMouseButtonCallback(globalGLFWindow, mouseButtonFunc);
		glfwSetCursorPosCallback(globalGLFWindow, cursorPosFunc);

		// create shader
		FSDraw = glCreateProgram();
		GLuint s = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(s, 1, &PFSDrawSource, 0);
		glCompileShader(s);
		glAttachShader(FSDraw, s);
		glLinkProgram(FSDraw);

		// create texture
		glActiveTexture(GL_TEXTURE0);
		glGenTextures(1, &GLFrameBufferTexture);
		glBindTexture(GL_TEXTURE_2D, GLFrameBufferTexture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB, globalWidth, globalHeight, 0, GL_LUMINANCE, GL_FLOAT, 0);

		// initialize some OpenGL state (will not change)
		glDisable(GL_DEPTH_TEST);

		glUseProgram(FSDraw);
		glUniform1i(glGetUniformLocation(FSDraw, "input_tex"), 0);

		GLint dims[4];
		glGetIntegerv(GL_VIEWPORT, dims);
		const float BufInfo[4] = { float(dims[2]), float(dims[3]), 1.0f / float(dims[2]), 1.0f / float(dims[3]) };
		glUniform4fv(glGetUniformLocation(FSDraw, "BufInfo"), 1, BufInfo);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, GLFrameBufferTexture);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
	}

	virtual ~OpenGLInit() {
		glfwTerminate();
	}
};



// main window
// you probably do not need to modify this in A0 to A3.
class CS488Window {
public:
	// put this first to make sure that the glInit's constructor is called before the one for CS488Window
	OpenGLInit glInit;

	CS488Window() {}
	virtual ~CS488Window() {}

	void(*process)() = NULL;

	void start() const {
		if (globalEnableParticles) {
			globalScene.addObject(&globalParticleSystem.particlesMesh);
			globalScene.addObject(&globalParticleSystem.shape);
		}
		globalScene.preCalc();

		// main loop
		while (glfwWindowShouldClose(globalGLFWindow) == GL_FALSE) {
			glfwPollEvents();
			globalViewDir = normalize(globalLookat - globalEye);
			globalRight = normalize(cross(globalViewDir, globalUp));

			if (globalEnableParticles) {
				globalParticleSystem.step();
			}

			if (globalRenderType == RENDER_RASTERIZE) {
				globalScene.Rasterize();
			} else if (globalRenderType == RENDER_RAYTRACE) {
				globalScene.Raytrace();
			} else if (globalRenderType == RENDER_IMAGE) {
				if (process) process();
			}

			if (globalRecording) {
				unsigned char* buf = new unsigned char[FrameBuffer.width * FrameBuffer.height * 4];
				int k = 0;
				for (int j = FrameBuffer.height - 1; j >= 0; j--) {
					for (int i = 0; i < FrameBuffer.width; i++) {
						buf[k++] = (unsigned char)(255.0f * Image::toneMapping(FrameBuffer.pixel(i, j).x));
						buf[k++] = (unsigned char)(255.0f * Image::toneMapping(FrameBuffer.pixel(i, j).y));
						buf[k++] = (unsigned char)(255.0f * Image::toneMapping(FrameBuffer.pixel(i, j).z));
						buf[k++] = 255;
					}
				}
				GifWriteFrame(&globalGIFfile, buf, globalWidth, globalHeight, globalGIFdelay);
				delete[] buf;
			}

			// drawing the frame buffer via OpenGL (you don't need to touch this)
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, globalWidth, globalHeight, GL_RGB, GL_FLOAT, &FrameBuffer.pixels[0][0]);
			glRecti(1, 1, -1, -1);
			glfwSwapBuffers(globalGLFWindow);
			globalFrameCount++;
			PCG32::rand();
		}
	}
};


