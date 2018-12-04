//The drawn potential should come from the same function being used in calculation.
 
#define FREEGLUT_STATIC
#define GLEW_STATIC
 
#define GL_GLEXT_PROTOTYPES
#define __CUDACC__
 
#include <Windows.h>
#include <GL\glew.h>
#include <GL\freeglut.h>
 
#include <CUDA\cuda.h>
#include <CUDA\cuda_runtime.h>
#include <CUDA\cuda_gl_interop.h>
 
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <GL\glm\glm.hpp>
 
//Delete data files in between runs.
//#define OUTPUT_DATA
 
#define L 8
// Best if GRID_SIZE % THREADS_PER_BLOCK == 0, though it might still work.
#define GRID_SIZE 1024
// THREADS_PER_BLOCK % MOD_DIVISOR == 0.
#define THREADS_PER_BLOCK 64
#define MOD_DIVISOR 8
#define PI 3.141592653589793
#define TIME_STEP 2 * PI / 128.
 
//Thin potential.
//#define X2_COEFFICIENT 40/2.
//#define X4_COEFFICIENT 20/2.
//#define POT_CONST 20/2.
 
//For middling waves.
//#define X2_COEFFICIENT 40/2.
//#define X4_COEFFICIENT 5/2.
//#define POT_CONST 80/2.
 
float x2Coefficient;
float x4Coefficient;
float potConst;
float minima;
 
float *devX2Coefficient;
float *devX4Coefficient;
float *devPotConst;
float *devMinima;
 
int parentWindow;
int timeWindow;
int mainVisWindow;
int avgPosWindow;
int avgMomentumWindow;
int avgEnergyWindow;
float *potential;
 
unsigned int pause = 0;
unsigned int tailLength = 0;
 
struct ComplexNum
{
	float Re;
	float Im;
 
	__device__ ComplexNum(void)
		:Re(0), Im(0) {      }
	__device__ ComplexNum(float Re_param, float Im_param)
		:Re(Re_param), Im(Im_param) {      }
	__device__ ComplexNum(const ComplexNum& Other)
		:Re(Other.Re), Im(Other.Im) {      }
 
	__device__ float MagnitudeSquared(void) const
	{ return Re*Re + Im*Im; }
 
	__device__ ComplexNum Conjugate(void) const
	{ return ComplexNum(Re, -Im); }
 
	__device__ ComplexNum SquareRoot(void) const
	{ return ComplexNum(sqrt(sqrt(this->MagnitudeSquared()) + Re)/sqrt(2.), glm::sign(Im) * sqrt(sqrt(this->MagnitudeSquared()) - Re)/sqrt(2.)); }
 
	__device__ ComplexNum operator*(const ComplexNum& Right) const
	{ return ComplexNum(Re * Right.Re - Im * Right.Im, Re * Right.Im + Im * Right.Re); }
 
	__device__ ComplexNum operator*(const float Right) const
	{ return ComplexNum(Re * Right, Im * Right); }
 
	__device__ ComplexNum operator/(const ComplexNum& Right)
	{ return ComplexNum(((*this) * Right.Conjugate())/Right.MagnitudeSquared()); }
 
	__device__ ComplexNum operator/(const float Right) const
	{ return ComplexNum(Re / Right, Im / Right); }
 
	__device__ ComplexNum operator+(const ComplexNum& Right) const
	{ return ComplexNum(Re + Right.Re, Im + Right.Im); }
 
	__device__ ComplexNum operator-(const ComplexNum& Right) const
	{ return ComplexNum(Re - Right.Re, Im - Right.Im); }
 
	__device__ ComplexNum operator=(const ComplexNum& Right)
	{ Re = Right.Re; Im = Right.Im; return *this; }
 
	__device__ bool operator!=(const ComplexNum& Right)
	{
		if(Re != Right.Re || Im != Right.Im)
			return true;
		else
			return false;
	}
 
	__device__ bool operator==(const ComplexNum& Right)
	{
		if(Re == Right.Re && Im == Right.Im)
			return true;
		else
			return false;
	}
};
 
__device__ ComplexNum operator/(const float Left, const ComplexNum& Right)
{
	return ComplexNum((Right.Conjugate() * Left) / Right.MagnitudeSquared());
}
 
struct VertColor
{
	float x, y, z;
	unsigned char r, g, b, a;
};
 
int parentWidth;
int parentHeight;
ComplexNum *devWaveFunctionIn;
ComplexNum *devWaveFunctionOut;
float *devSumHolder;
 
class CudaGLInterop //Needs destructor.
{
	const unsigned int WIDTH;
	const unsigned int HEIGHT;
	unsigned int sharedBuf;
	unsigned int sharedTex;
	cudaGraphicsResource *resource;
	void *devPtr;
	size_t size;
	const GLenum TYPE;
public:
	CudaGLInterop(int width, int height, GLenum type);
	void * MapResource(void);
	void UnmapResource(void);
	void Draw(GLenum drawMode, int end) const;
	//remember to set up context for line smoothing
	void DrawPixAsTex(void) const;
};
 
__global__ void AveragePos(ComplexNum const * const waveFunction, VertColor * const vertexBuf, int * const timeIndex);
__global__ void AverageMomentum(ComplexNum const * const waveFunction, VertColor * const vertexBuf, int * const timeIndex);
__global__ void AverageKEnergy(ComplexNum const * const waveFunction, VertColor * const vertexBuf, int * const timeIndex);
__global__ void AverageEnergy(ComplexNum const * const waveFunction, VertColor * const vertexBuf, int * const timeIndex, float const * const devX4Coefficient, float const * const devX2Coefficient, float const * const devPotConst);
void CalculateCoefficients(void);
inline void CudaError(int lineNum);
float* CreatePotential(void);
void DebugDevData(void);
void DebugVertArray(VertColor const * const devPtr);
void DisplayMainVis(void);
void DisplayParent(void);
void DisplayPotential(void);
void DisplayAvgPos(void);
void DisplayAvgMomentum(void);
void DisplayAvgEnergy(void);
void DisplayAvgKEnergy(void);
void DrawGrid(int const * const xRange, const float xStep, int const * const yRange, const float yStep);
void DrawString(float x_pos, float y_pos, float x_scale, float y_scale, float z_scale, void *font, char *string);
void DrawTime(float x_pos, float y_pos, float x_scale, float y_scale, float z_scale, void *font, unsigned int frames);
inline void GLError(int lineNum);
__global__ void FillDrawBuffers(ComplexNum const * const waveFunction, uchar4 * const pixelBuf, VertColor * const vertexBuf);
__global__ void InitializeWaveFuncAtomic(ComplexNum * const waveFunction, float * const sumHolder, float const * const devMinima);
__device__ ComplexNum KMatrix(const int end, const int start, float const * const devX4Coefficient, float const * const devX2Coefficient, float const * const devPotConst);
__global__ void NormalizeWaveFunction(ComplexNum const * const waveFunctionIn, float * const sumHolder);
__global__ void NormalizeWaveFunction2(ComplexNum const * const waveFunctionIn, ComplexNum * const waveFunctionOut, float const * const sumHolder);
__global__ void PropogateWaveFunction(ComplexNum const * const waveFunctionIn, ComplexNum * const waveFunctionOut, float const * const devX4Coefficient, float const * const devX2Coefficient, float const * const devPotConst);
void ReshapeParent(int width, int height);
void ReshapeMain(int width, int height);
void ReshapePos(int width, int height);
void ReshapeMomentum(int width, int height);
void ReshapeEnergy(int width, int height);
void Keys(unsigned char key, int x, int y);
void Timer(int unused);
__device__ ComplexNum WaveFunction(const int index, float const * const devMinima);
 
int main(int argc, char ** argv)
{
	CalculateCoefficients();
 
	cudaDeviceProp devProp;
	memset((void*)&devProp, 0, sizeof(cudaDeviceProp));
	devProp.major = 1.0;
	devProp.minor = 0.0;
	int devChoice;
	cudaChooseDevice(&devChoice, &devProp);
	cudaSetDevice(devChoice);
	cudaGLSetGLDevice(devChoice);
 
	CudaError(__LINE__);
 
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA);
 
	//int initWidth = 1024, initHeight = 576;
	int initWidth = 1920, initHeight = 1080;
	glutInitWindowSize(initWidth, initHeight);
 
	parentWindow = glutCreateWindow("Feynman Path Integral Visualisation");
	glutFullScreen();
	glClearColor(0,0,0,1);
	glutReshapeFunc(ReshapeParent);
	//glutDisplayFunc(DisplayParent);
	glutKeyboardFunc(Keys);
 
	timeWindow = glutCreateSubWindow(parentWindow,0, initHeight / 8, initWidth / 6, initHeight / 8);
	glClearColor(0,0,0,1);
	glEnable(GL_LINE_SMOOTH);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
	glLineWidth(2);
	glOrtho(-1,1,-1,1,0,1);
 
	mainVisWindow = glutCreateSubWindow(parentWindow, 0, 0, 3 * initWidth / 4, initHeight);
	glutReshapeFunc(ReshapeMain);
	//glutDisplayFunc(DisplayMainVis);
	glClearColor(0,0,0,1);
	glEnable(GL_LINE_SMOOTH);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
	glLineWidth(2);
	glOrtho(-1,1,-.25,2,0,1);
 
	avgPosWindow = glutCreateSubWindow(parentWindow, 3 * initWidth / 4, 0, initWidth / 4, initHeight/3);
	glutReshapeFunc(ReshapePos);
	//glutDisplayFunc(DisplayAvgPos);
	glClearColor(0,0,0,1);
	glEnable(GL_LINE_SMOOTH);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
	glLineWidth(2);
	glPointSize(5);
	glOrtho(0,1024,-1.5,1.5,0,1);
 
	avgMomentumWindow = glutCreateSubWindow(parentWindow, 3 * initWidth / 4, initHeight/3, initWidth / 4, initHeight/3);
	glutReshapeFunc(ReshapeMomentum);
	//glutDisplayFunc(DisplayAvgMomentum);
	glClearColor(0,0,0,1);
	glEnable(GL_LINE_SMOOTH);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
	glLineWidth(2);
	glPointSize(5);
	glOrtho(0,1024,-1.5,1.5,0,1);
 
	avgEnergyWindow = glutCreateSubWindow(parentWindow, 3 * initWidth / 4,  2 * initHeight/3, initWidth / 4, initHeight/3);
	glutReshapeFunc(ReshapeEnergy);
	//glutDisplayFunc(DisplayAvgEnergy);
	glClearColor(0,0,0,1);
	glEnable(GL_LINE_SMOOTH);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
	glLineWidth(2);
	glPointSize(5);
	glOrtho(0,1024,0,7.5,0,1);
 
	GLenum error = glewInit();
	if(error != GLEW_OK)
	{
		std::cout << "GLEW!!!";
		exit(1);
	}
 
	cudaMalloc((void**) &devX4Coefficient, sizeof(float));
	cudaMemcpy(devX4Coefficient, &x4Coefficient, sizeof(float), cudaMemcpyHostToDevice);
 
	cudaMalloc((void**) &devX2Coefficient, sizeof(float));
	cudaMemcpy(devX2Coefficient, &x2Coefficient, sizeof(float), cudaMemcpyHostToDevice);
 
	cudaMalloc((void**) &devPotConst, sizeof(float));
	cudaMemcpy(devPotConst, &potConst, sizeof(float), cudaMemcpyHostToDevice);
 
	cudaMalloc((void**) &devMinima, sizeof(float));
	cudaMemcpy(devMinima, &minima, sizeof(float), cudaMemcpyHostToDevice);
 
	CudaError(__LINE__);
 
	potential = CreatePotential();
 
	cudaMalloc((void**) &devWaveFunctionIn, GRID_SIZE*sizeof(ComplexNum));
	cudaMemset(devWaveFunctionIn, 0, GRID_SIZE*sizeof(ComplexNum));
	CudaError(__LINE__);
 
	cudaMalloc((void**) &devWaveFunctionOut, GRID_SIZE*sizeof(ComplexNum));
	cudaMemset(devWaveFunctionOut, 0, GRID_SIZE*sizeof(ComplexNum));
	CudaError(__LINE__);
 
	cudaMalloc((void**) &devSumHolder, sizeof(float));
	cudaMemset(devSumHolder, 0, sizeof(float));
	CudaError(__LINE__);
 
	dim3 blocks((GRID_SIZE + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK);
	dim3 threads(THREADS_PER_BLOCK);
 
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
 
	InitializeWaveFuncAtomic<<<blocks, threads>>>(devWaveFunctionIn, devSumHolder, devMinima);
	NormalizeWaveFunction2<<<blocks.x, threads>>>(devWaveFunctionIn, devWaveFunctionOut, devSumHolder);
 
	cudaMemcpy(devWaveFunctionIn, devWaveFunctionOut, GRID_SIZE * sizeof(ComplexNum), cudaMemcpyDeviceToDevice);
	cudaMemset(devWaveFunctionOut, 0, GRID_SIZE * sizeof(ComplexNum));
 
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);
	std::cout << "Initialization Time (Atomic): " << time << "ms" << std::endl;
 
	CudaError(__LINE__);
 
	DebugDevData();
 
	Timer(0);
	glutMainLoop();
}
 
 
__global__ void AveragePos(ComplexNum const * const waveFunction, VertColor * const vertexBuf, int * const timeIndex)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ ComplexNum rowSum[THREADS_PER_BLOCK];
	float deltaX = (float) L/GRID_SIZE;
 
	rowSum[threadIdx.x] = waveFunction[index].Conjugate() * ((float) index * deltaX - (float) L/2.) * waveFunction[index] * deltaX;
 
	int Mod = THREADS_PER_BLOCK / MOD_DIVISOR;
 
	if(threadIdx.x % Mod != 0)
	{
		atomicAdd(&rowSum[threadIdx.x - threadIdx.x % Mod].Re, rowSum[threadIdx.x].Re);
		atomicAdd(&rowSum[threadIdx.x - threadIdx.x % Mod].Im, rowSum[threadIdx.x].Im);
	}
	else if(threadIdx.x < Mod && threadIdx.x != 0)
	{
		atomicAdd(&rowSum[0].Re, rowSum[threadIdx.x].Re);
		atomicAdd(&rowSum[0].Im, rowSum[threadIdx.x].Im);
	}
 
	__syncthreads();
 
	if(threadIdx.x % Mod == 0 && threadIdx.x >= Mod)
	{
		atomicAdd(&rowSum[0].Re, rowSum[threadIdx.x].Re);
		atomicAdd(&rowSum[0].Im, rowSum[threadIdx.x].Im);
	}
 
	__syncthreads();
 
	if(index == 0)
	{
		vertexBuf[*timeIndex].x = (*timeIndex);
		vertexBuf[*timeIndex].g = 100;
		vertexBuf[*timeIndex].b = 200;
		vertexBuf[*timeIndex].a = 200;
	}
 
	if(threadIdx.x == 0)
		atomicAdd(&vertexBuf[*timeIndex].y, rowSum[0].Re);
 
	//vertexBuf[index].x = (float) ((float) index * L/GRID_SIZE - L/2.)/(L/2.);
	//vertexBuf[index].y = *timeIndex;
	//vertexBuf[index].g = 100;
	//vertexBuf[index].b = 200;
	//vertexBuf[index].a = 200;
}
__global__ void AverageMomentum(ComplexNum const * const waveFunction, VertColor * const vertexBuf, int * const timeIndex)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ ComplexNum rowSum[THREADS_PER_BLOCK];
	float deltaX = (float) L/GRID_SIZE;
 
	if(threadIdx.x < blockDim.x - 1)
		rowSum[threadIdx.x] = waveFunction[index].Conjugate() * ComplexNum(0, -1) * (waveFunction[index + 1] - waveFunction[index])* (.5/deltaX) * deltaX;
	else
		rowSum[threadIdx.x] = ComplexNum(0, 0);
 
	int Mod = THREADS_PER_BLOCK / MOD_DIVISOR;
 
	if(threadIdx.x % Mod != 0)
	{
		atomicAdd(&rowSum[threadIdx.x - threadIdx.x % Mod].Re, rowSum[threadIdx.x].Re);
		atomicAdd(&rowSum[threadIdx.x - threadIdx.x % Mod].Im, rowSum[threadIdx.x].Im);
	}
	else if(threadIdx.x < Mod && threadIdx.x != 0)
	{
		atomicAdd(&rowSum[0].Re, rowSum[threadIdx.x].Re);
		atomicAdd(&rowSum[0].Im, rowSum[threadIdx.x].Im);
	}
 
	__syncthreads();
 
	if(threadIdx.x % Mod == 0 && threadIdx.x >= Mod)
	{
		atomicAdd(&rowSum[0].Re, rowSum[threadIdx.x].Re);
		atomicAdd(&rowSum[0].Im, rowSum[threadIdx.x].Im);
	}
 
	__syncthreads();
 
	if(index == 0)
	{
		vertexBuf[*timeIndex].x = (*timeIndex);
		vertexBuf[*timeIndex].g = 100;
		vertexBuf[*timeIndex].b = 200;
		vertexBuf[*timeIndex].a = 200;
	}
 
	if(threadIdx.x == 0)
		atomicAdd(&vertexBuf[*timeIndex].y, rowSum[0].Re);
 
	//vertexBuf[index].x = (float) ((float) index * L/GRID_SIZE - L/2.)/(L/2.);
	//vertexBuf[index].y = *timeIndex;
	//vertexBuf[index].g = 100;
	//vertexBuf[index].b = 200;
	//vertexBuf[index].a = 200;
}
__global__ void AverageKEnergy(ComplexNum const * const waveFunction, VertColor * const vertexBuf, int * const timeIndex)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ ComplexNum rowSum[THREADS_PER_BLOCK];
	float deltaX = (float) L/GRID_SIZE;
 
	if(threadIdx.x < blockDim.x - 2)
		rowSum[threadIdx.x] = waveFunction[index].Conjugate() * (waveFunction[index + 2] - waveFunction[index + 1] * 2 + waveFunction[index]) *  (-.5/pow(deltaX, 2)) * deltaX;
	else
		rowSum[threadIdx.x] = ComplexNum(0, 0);
 
	int Mod = THREADS_PER_BLOCK / MOD_DIVISOR;
 
	if(threadIdx.x % Mod != 0)
	{
		atomicAdd(&rowSum[threadIdx.x - threadIdx.x % Mod].Re, rowSum[threadIdx.x].Re);
		atomicAdd(&rowSum[threadIdx.x - threadIdx.x % Mod].Im, rowSum[threadIdx.x].Im);
	}
	else if(threadIdx.x < Mod && threadIdx.x != 0)
	{
		atomicAdd(&rowSum[0].Re, rowSum[threadIdx.x].Re);
		atomicAdd(&rowSum[0].Im, rowSum[threadIdx.x].Im);
	}
 
	__syncthreads();
 
	if(threadIdx.x % Mod == 0 && threadIdx.x >= Mod)
	{
		atomicAdd(&rowSum[0].Re, rowSum[threadIdx.x].Re);
		atomicAdd(&rowSum[0].Im, rowSum[threadIdx.x].Im);
	}
 
	__syncthreads();
 
	if(index == 0)
	{
		vertexBuf[*timeIndex].x = (*timeIndex);
		vertexBuf[*timeIndex].g = 100;
		vertexBuf[*timeIndex].b = 200;
		vertexBuf[*timeIndex].a = 200;
	}
 
	if(threadIdx.x == 0)
		atomicAdd(&vertexBuf[*timeIndex].y, rowSum[0].Re);
 
	//vertexBuf[index].x = (float) ((float) index * L/GRID_SIZE - L/2.)/(L/2.);
	//vertexBuf[index].y = *timeIndex;
	//vertexBuf[index].g = 100;
	//vertexBuf[index].b = 200;
	//vertexBuf[index].a = 200;
}
__global__ void AverageEnergy(ComplexNum const * const waveFunction, VertColor * const vertexBuf, int * const timeIndex, float const * const devX4Coefficient, float const * const devX2Coefficient, float const * const devPotConst)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ ComplexNum rowSum[THREADS_PER_BLOCK];
	float deltaX = (float) L/GRID_SIZE;
 
	if(threadIdx.x < blockDim.x - 2)
		rowSum[threadIdx.x] = waveFunction[index + 1].Conjugate() * (waveFunction[index + 2] - waveFunction[index + 1] * 2 + waveFunction[index]) *  (-.5/pow(deltaX, 2)) * deltaX + waveFunction[index + 1].Conjugate() * waveFunction[index + 1] * ( *devX4Coefficient * pow((index + 1) * deltaX - L/2., 4) - *devX2Coefficient * pow((index + 1) * deltaX - L/2., 2) + *devPotConst ) * deltaX;
 
	else
		rowSum[threadIdx.x] = ComplexNum(0, 0);
 
	int Mod = THREADS_PER_BLOCK / MOD_DIVISOR;
 
	if(threadIdx.x % Mod != 0)
	{
		atomicAdd(&rowSum[threadIdx.x - threadIdx.x % Mod].Re, rowSum[threadIdx.x].Re);
		atomicAdd(&rowSum[threadIdx.x - threadIdx.x % Mod].Im, rowSum[threadIdx.x].Im);
	}
	else if(threadIdx.x < Mod && threadIdx.x != 0)
	{
		atomicAdd(&rowSum[0].Re, rowSum[threadIdx.x].Re);
		atomicAdd(&rowSum[0].Im, rowSum[threadIdx.x].Im);
	}
 
	__syncthreads();
 
	if(threadIdx.x % Mod == 0 && threadIdx.x >= Mod)
	{
		atomicAdd(&rowSum[0].Re, rowSum[threadIdx.x].Re);
		atomicAdd(&rowSum[0].Im, rowSum[threadIdx.x].Im);
	}
 
	__syncthreads();
 
	if(index == 0)
	{
		vertexBuf[*timeIndex].x = (*timeIndex);
		vertexBuf[*timeIndex].g = 100;
		vertexBuf[*timeIndex].b = 200;
		vertexBuf[*timeIndex].a = 200;
	}
 
	if(threadIdx.x == 0)
		atomicAdd(&vertexBuf[*timeIndex].y, rowSum[0].Re);
 
	//vertexBuf[index].x = (float) ((float) index * L/GRID_SIZE - L/2.)/(L/2.);
	//vertexBuf[index].y = *timeIndex;
	//vertexBuf[index].g = 100;
	//vertexBuf[index].b = 200;
	//vertexBuf[index].a = 200;
}
void CalculateCoefficients(void)
//The negative on the x2Coefficient is coded elsewhere.
{
	std::cout << "Specify the constant parameter (height, width): ";
 
	std::string constParameter;
	std::cin >> constParameter;
 
	float beta;
 
	if(constParameter[0] != 'm')
	{
		std::cout << "Enter beta value: ";
		std::cin >> beta;
	}
 
	switch(constParameter[0])
	{
		case 'h':
		case 'H':
			x4Coefficient = pow(beta, 2) / 4.;
			x2Coefficient = beta;
			potConst = 1;
			minima = sqrt(2./beta);
		break;
 
		case 'w':
		case 'W':
			x4Coefficient = beta / 2.;
			x2Coefficient = beta;
			potConst = beta / 2.;
			minima = 1;
		break;
 
		case 'm':
			std::cout << "x4Coefficient: ";
			std::cin >> x4Coefficient;
 
			std::cout << "x2Coefficient: ";
			std::cin >> x2Coefficient;
			x2Coefficient = -x2Coefficient;
 
			std::cout << "potConst: ";
			std::cin >> potConst;
 
			std::cout << "x0: ";
			std::cin >> minima;
		break;
 
		default:
			std::cout << "Invalid Selection" << std::endl;
			exit(1);
		break;
	}
}
inline void CudaError(int lineNum)
{
	cudaError_t cudaErr;
	cudaErr = cudaGetLastError();
	if(cudaErr != CUDA_SUCCESS)
	{
		std::cout << "CUDA error: " << cudaGetErrorString(cudaErr) << " at " << lineNum << " in " << __FILE__ << std::endl;
		exit(1);
	}
}
float* CreatePotential(void)
{
	float deltaX = (float) L/GRID_SIZE;
 
	float *potential = new float[GRID_SIZE];
 
	for(int i = 0; i < GRID_SIZE; ++i)
		potential[i] = (x4Coefficient * pow(float(i * deltaX - L/2.),(float)4.) - x2Coefficient * pow(float(i * deltaX - L/2.),(float)2.) + potConst)/potConst;
 
	return potential;
}
void DebugDevData(void)
{
	static ComplexNum *hostPtr = new ComplexNum[GRID_SIZE];
	cudaMemcpy(hostPtr, devWaveFunctionIn, GRID_SIZE * sizeof(ComplexNum), cudaMemcpyDeviceToHost);
	CudaError(__LINE__);
}
void DebugVertArray(VertColor const * const devPtr)
{
	static VertColor *hostPtr = new VertColor[GRID_SIZE];
	cudaMemcpy(hostPtr, devPtr, GRID_SIZE*sizeof(VertColor), cudaMemcpyDeviceToHost);
	CudaError(__LINE__);
}
void DisplayMainVis(void)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float time;
	static bool firstTime = true;
 
	static CudaGLInterop pixelResource(GRID_SIZE, GRID_SIZE, GL_PIXEL_UNPACK_BUFFER);
	static CudaGLInterop vertexResource(GRID_SIZE, 1, GL_ARRAY_BUFFER);
 
	uchar4 *devPixelPtr = (uchar4*) pixelResource.MapResource();
	VertColor *devVertexPtr = (VertColor*) vertexResource.MapResource();
 
	dim3 blocks((GRID_SIZE + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, GRID_SIZE);
	dim3 threads(THREADS_PER_BLOCK);
 
	if(firstTime == true)
		cudaEventRecord(start);
	FillDrawBuffers<<<blocks.x, threads>>>(devWaveFunctionIn, devPixelPtr, devVertexPtr);
	if(firstTime == true)
	{
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);	
		cudaEventElapsedTime(&time, start, stop);
		std::cout << "Fill Buffer Time: " << time << "ms" << std::endl;
	}
 
	if(firstTime == true)
		cudaEventRecord(start);
	PropogateWaveFunction<<<blocks, threads>>>(devWaveFunctionIn, devWaveFunctionOut, devX4Coefficient, devX2Coefficient, devPotConst);
	if(firstTime == true)
	{
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);	
		cudaEventElapsedTime(&time, start, stop);
		std::cout << "Propogation Time: " << time << "ms" << std::endl;
	}
	firstTime = false;
 
	CudaError(__LINE__);
 
	pixelResource.UnmapResource();
	vertexResource.UnmapResource();
 
	pixelResource.DrawPixAsTex();
	vertexResource.Draw(GL_LINE_STRIP, GRID_SIZE);
 
	cudaMemcpy(devWaveFunctionIn, devWaveFunctionOut, GRID_SIZE * sizeof(ComplexNum), cudaMemcpyDeviceToDevice);
	cudaMemset(devWaveFunctionOut, 0, GRID_SIZE * sizeof(ComplexNum));
 
	cudaMemset(devSumHolder, 0, sizeof(float));
 
	NormalizeWaveFunction<<<blocks.x, threads>>>(devWaveFunctionIn, devSumHolder);
	NormalizeWaveFunction2<<<blocks.x, threads>>>(devWaveFunctionIn, devWaveFunctionOut, devSumHolder);
 
	cudaMemcpy(devWaveFunctionIn, devWaveFunctionOut, GRID_SIZE * sizeof(ComplexNum), cudaMemcpyDeviceToDevice);
	cudaMemset(devWaveFunctionOut, 0, GRID_SIZE * sizeof(ComplexNum));
 
	CudaError(__LINE__);
}
void DisplayParent(void)
{
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
 
}
void DisplayPotential(void)
{
	float deltaX = (float) L/GRID_SIZE;
 
	glBegin(GL_LINE_STRIP);
		for(int i = 0; i < GRID_SIZE; ++i)
			glVertex3f(float((i * deltaX - L/2.)/(L/2.)), potential[i], 0);
	glEnd();
}
void DisplayAvgPos(void)
{
	static VertColor *avgPos = new VertColor[GRID_SIZE];
 
	static int *timeIndex = new int(0);
	int * devTimeIndex;
	cudaMalloc((void **) &devTimeIndex, sizeof(int));
 
	if(*timeIndex == 0 || *timeIndex == GRID_SIZE)
	{
		memset(avgPos, 0, GRID_SIZE * sizeof(VertColor));
		*timeIndex = 0;
	}
 
	cudaMemcpy(devTimeIndex, timeIndex, sizeof(int), cudaMemcpyHostToDevice); 
 
	static CudaGLInterop vertexResource(GRID_SIZE, 1, GL_ARRAY_BUFFER);
 
	dim3 blocks((GRID_SIZE + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, GRID_SIZE);
	dim3 threads(THREADS_PER_BLOCK);
 
	VertColor *devVertexPtr = (VertColor*) vertexResource.MapResource();
 
	cudaMemcpy(devVertexPtr, avgPos, GRID_SIZE * sizeof(VertColor), cudaMemcpyHostToDevice);
 
	AveragePos<<<blocks.x, threads>>>(devWaveFunctionIn, devVertexPtr, devTimeIndex);
 
	cudaMemcpy(avgPos, devVertexPtr, GRID_SIZE * sizeof(VertColor), cudaMemcpyDeviceToHost);
 
	DebugVertArray(devVertexPtr);
 
	vertexResource.UnmapResource();
 
	#ifdef OUTPUT_DATA
		std::fstream fout("Data Files/PositionData.data", std::fstream::out | std::fstream::app);
		fout << avgPos[*timeIndex].x * TIME_STEP << " " << avgPos[*timeIndex].y << std::endl;
		fout.close();
	#endif
 
	++(*timeIndex);
	cudaFree(devTimeIndex);
 
	vertexResource.Draw(GL_LINE_STRIP, *timeIndex);
}
void DisplayAvgMomentum(void)
{
	static VertColor *avgMomentum = new VertColor[GRID_SIZE];
 
	static int *timeIndex = new int(0);
	int * devTimeIndex;
	cudaMalloc((void **) &devTimeIndex, sizeof(int));
 
	if(*timeIndex == 0 || *timeIndex == GRID_SIZE)
	{
		memset(avgMomentum, 0, GRID_SIZE * sizeof(VertColor));
		*timeIndex = 0;
	}
 
	cudaMemcpy(devTimeIndex, timeIndex, sizeof(int), cudaMemcpyHostToDevice); 
 
	static CudaGLInterop vertexResource(GRID_SIZE, 1, GL_ARRAY_BUFFER);
 
	dim3 blocks((GRID_SIZE + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, GRID_SIZE);
	dim3 threads(THREADS_PER_BLOCK);
 
	VertColor *devVertexPtr = (VertColor*) vertexResource.MapResource();
 
	cudaMemcpy(devVertexPtr, avgMomentum, GRID_SIZE * sizeof(VertColor), cudaMemcpyHostToDevice);
 
	AverageMomentum<<<blocks.x, threads>>>(devWaveFunctionIn, devVertexPtr, devTimeIndex);
 
	cudaMemcpy(avgMomentum, devVertexPtr, GRID_SIZE * sizeof(VertColor), cudaMemcpyDeviceToHost);
 
	DebugVertArray(devVertexPtr);
 
	vertexResource.UnmapResource();
 
	#ifdef OUTPUT_DATA
		std::fstream fout("Data Files/MomentumData.data", std::fstream::out | std::fstream::app);
		fout << avgMomentum[*timeIndex].x * TIME_STEP << " " << avgMomentum[*timeIndex].y << std::endl;
		fout.close();
	#endif
 
	++(*timeIndex);
	cudaFree(devTimeIndex);
 
	vertexResource.Draw(GL_LINE_STRIP, *timeIndex);	
}
void DisplayAvgEnergy(void)
{
	static VertColor *avgEnergy = new VertColor[GRID_SIZE];
 
	static int *timeIndex = new int(0);
	int * devTimeIndex;
	cudaMalloc((void **) &devTimeIndex, sizeof(int));
 
	if(*timeIndex == 0 || *timeIndex == GRID_SIZE)
	{
		memset(avgEnergy, 0, GRID_SIZE * sizeof(VertColor));
		*timeIndex = 0;
	}
 
	cudaMemcpy(devTimeIndex, timeIndex, sizeof(int), cudaMemcpyHostToDevice); 
 
	static CudaGLInterop vertexResource(GRID_SIZE, 1, GL_ARRAY_BUFFER);
 
	dim3 blocks((GRID_SIZE + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, GRID_SIZE);
	dim3 threads(THREADS_PER_BLOCK);
 
	VertColor *devVertexPtr = (VertColor*) vertexResource.MapResource();
 
	cudaMemcpy(devVertexPtr, avgEnergy, GRID_SIZE * sizeof(VertColor), cudaMemcpyHostToDevice);
 
	AverageEnergy<<<blocks.x, threads>>>(devWaveFunctionIn, devVertexPtr, devTimeIndex, devX4Coefficient, devX2Coefficient, devPotConst);
 
	cudaMemcpy(avgEnergy, devVertexPtr, GRID_SIZE * sizeof(VertColor), cudaMemcpyDeviceToHost);
 
	DebugVertArray(devVertexPtr);
 
	vertexResource.UnmapResource();
 
	#ifdef OUTPUT_DATA
		std::fstream fout("Data Files/EnergyData.data", std::fstream::out | std::fstream::app);
		fout << avgEnergy[*timeIndex].x * TIME_STEP << " " << avgEnergy[*timeIndex].y << std::endl;
		fout.close();
	#endif
 
	++(*timeIndex);
	cudaFree(devTimeIndex);
 
	vertexResource.Draw(GL_LINE_STRIP, *timeIndex);
}
void DisplayAvgKEnergy(void)
{
	static VertColor *avgEnergy = new VertColor[GRID_SIZE];
 
	static int *timeIndex = new int(0);
	int * devTimeIndex;
	cudaMalloc((void **) &devTimeIndex, sizeof(int));
 
	if(*timeIndex == 0 || *timeIndex == GRID_SIZE)
	{
		memset(avgEnergy, 0, GRID_SIZE * sizeof(VertColor));
		*timeIndex = 0;
	}
 
	cudaMemcpy(devTimeIndex, timeIndex, sizeof(int), cudaMemcpyHostToDevice); 
 
	static CudaGLInterop vertexResource(GRID_SIZE, 1, GL_ARRAY_BUFFER);
 
	dim3 blocks((GRID_SIZE + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, GRID_SIZE);
	dim3 threads(THREADS_PER_BLOCK);
 
	VertColor *devVertexPtr = (VertColor*) vertexResource.MapResource();
 
	cudaMemcpy(devVertexPtr, avgEnergy, GRID_SIZE * sizeof(VertColor), cudaMemcpyHostToDevice);
 
	AverageKEnergy<<<blocks.x, threads>>>(devWaveFunctionIn, devVertexPtr, devTimeIndex);
 
	cudaMemcpy(avgEnergy, devVertexPtr, GRID_SIZE * sizeof(VertColor), cudaMemcpyDeviceToHost);
 
	DebugVertArray(devVertexPtr);
 
	vertexResource.UnmapResource();
 
	#ifdef OUTPUT_DATA
		std::fstream fout("Data Files/KEnergyData.data", std::fstream::out | std::fstream::app);
		fout << avgEnergy[*timeIndex].x * TIME_STEP<< " " << avgEnergy[*timeIndex].y << std::endl;
		fout.close();
	#endif
 
	++(*timeIndex);
	cudaFree(devTimeIndex);
 
	vertexResource.Draw(GL_LINE_STRIP, *timeIndex);
}
void DrawGrid(int const * const xRange, const float xStep, int const * const yRange, const float yStep)
	//Ranges are arrays of 2, min and max.
{
	glBegin(GL_LINES);
		for(int i = 0; i < (int) (xRange[1] + abs(xRange[0]))/xStep; ++i)
			for(int j = 0; j < 2; ++j)
				glVertex2f(i * xStep + xRange[0], yRange[j]);
 
		for(int i = 0; i < (int) (yRange[1] + abs(yRange[0]))/yStep; ++i)
			for(int j = 0; j < 2; ++j)
				glVertex2f(xRange[j], i * yStep + yRange[0]);
	glEnd();
}
void DrawString(float x_pos, float y_pos, float x_scale, float y_scale, float z_scale, void *font, char *string)
{
	char *c;
	glPushMatrix();
	glTranslatef(x_pos, y_pos, 0);
	glScalef(x_scale, y_scale, z_scale);
	for(c = string; *c != '\0'; ++c)
	{
		glutStrokeCharacter(font, *c);
	}
	glPopMatrix();
}
void DrawTime(float x_pos, float y_pos, float x_scale, float y_scale, float z_scale, void *font, unsigned int frames)
{
	std::stringstream sstream;
 
	glPushMatrix();
	glTranslatef(x_pos, y_pos, 0);
	glScalef(x_scale, y_scale, z_scale);
 
	sstream << "Frames: " << frames << "\n";
	glutStrokeString(font, (const unsigned char*) (sstream.str()).c_str());
 
	float time = frames * TIME_STEP;
 
	sstream.str("");
	sstream << "Time: " << time;
	glutStrokeString(font, (const unsigned char*) (sstream.str()).c_str());
 
	glPopMatrix();
}
__global__ void FillDrawBuffers(ComplexNum const * const waveFunction, uchar4 * const pixelBuf, VertColor * const vertexBuf)
// One dimensional in blocks and threads.
// BGRA
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int iMax = 40;
 
	if(index < GRID_SIZE)
	{
		if(pixelBuf != NULL)
		{
			for(int i = 0; i < iMax; ++i)
			{
				int offset = GRID_SIZE*GRID_SIZE/2 + i*GRID_SIZE;
				int tempColor;
 
				tempColor = (float) waveFunction[index].MagnitudeSquared() * 255 * 1.3 * sin(PI * i / (iMax-1));
 
				if(tempColor < 256 && tempColor > -1)
					pixelBuf[index + offset].x = tempColor;
				else
					pixelBuf[index + offset].x = 255;
			}
		}
 
		if(vertexBuf != NULL)
		{
			vertexBuf[index].x = (float) ((float) index * L/GRID_SIZE - L/2.)/(L/2.);
			vertexBuf[index].y = waveFunction[index].MagnitudeSquared();
			vertexBuf[index].g = 100;
			vertexBuf[index].b = 200;
			vertexBuf[index].a = 200;
		}
	}
}
inline void GLError(int lineNum)
{
	GLenum glErr;
	glErr = glGetError();
	if(glErr != GL_NO_ERROR)
	{
		std::cout << "GL error: " << gluErrorString(glErr) << " at " << lineNum << " in " << __FILE__ << std::endl;
		exit(1);
	}
}
__global__ void InitializeWaveFuncAtomic(ComplexNum * const waveFunction, float * const sumHolder, float const * const devMinima)
//One dimensional in blocks and threads.
//Set waveFunction to zero before, using cudaMemset.
//Set sumHolder to zero before, using cudaMemset.
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	float deltaX = (float) L/GRID_SIZE;
	__shared__ float riemannTerms[THREADS_PER_BLOCK];
 
	waveFunction[index] = WaveFunction(index, devMinima);
 
	riemannTerms[threadIdx.x] = waveFunction[index].MagnitudeSquared() *  deltaX;	
 
	__syncthreads();
 
	int Mod = THREADS_PER_BLOCK / MOD_DIVISOR;
 
	if(threadIdx.x % Mod != 0)
		atomicAdd(&riemannTerms[threadIdx.x - threadIdx.x % Mod], riemannTerms[threadIdx.x]);
 
	__syncthreads();
 
	if(threadIdx.x % Mod == 0  && threadIdx.x != 0)
		atomicAdd(&riemannTerms[0], riemannTerms[threadIdx.x]);
 
	__syncthreads();
 
	if(threadIdx.x == 0)
		atomicAdd(sumHolder, riemannTerms[0]);
}
void Keys(unsigned char key, int x, int y)
{
	switch(key)
	{
		case ' ':
			if(pause < 1000000)
				pause = 1000000;
			else
				pause = 1000;
		break;
		case ',': 
			pause += 200;
		break;
		case '.':
			if(pause > 0 )
				pause -= 200;
			else
				pause = 0;
		break;
 
		case 27: //esc
			exit(0);
		break;
 
		case '+': 
			tailLength += 10;
		break;
		case '-':
			if(tailLength>0)
				tailLength -= 10;
			else
				tailLength = 0;
		break;
	}
}
__device__ ComplexNum KMatrix(const int end, const int start, float const * const devX4Coefficient, float const * const devX2Coefficient, float const * const devPotConst)
{
	float deltaT = (float) TIME_STEP;
	float deltaX = (float) L/GRID_SIZE;
 
	return ComplexNum(cos(pow((float) end * deltaX - (float) start * deltaX ,(float)2.)/(2. * deltaT) - *devX4Coefficient * pow((float) end * deltaX + (float) start * deltaX - L,(float)4.)/16. * deltaT + *devX2Coefficient * pow((float) end * deltaX + (float) start * deltaX - L,(float)2.)/4. * deltaT - *devPotConst), sin(pow((float) end * deltaX - (float) start * deltaX,(float)2.)/(2. * deltaT) - *devX4Coefficient * pow((float) end * deltaX + (float) start * deltaX - L,(float)4.)/16. * deltaT + *devX2Coefficient * pow((float) end * deltaX + (float) start * deltaX - L,(float)2.)/4. * deltaT - *devPotConst));
}
__global__ void NormalizeWaveFunction(ComplexNum const * const waveFunctionIn, float * const sumHolder)
//Set sumHolder to zero before entering function.
//Integrates the wave function. Normalization occurs in NormalizeWaveFunction2.
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ float rowSum[THREADS_PER_BLOCK];
	float deltaX = (float) L/GRID_SIZE;
 
	rowSum[threadIdx.x] = waveFunctionIn[index].MagnitudeSquared() * deltaX; 
 
	__syncthreads();
 
	int Mod = THREADS_PER_BLOCK / MOD_DIVISOR;
 
	if(threadIdx.x % Mod != 0)
		atomicAdd(&rowSum[threadIdx.x - threadIdx.x % Mod], rowSum[threadIdx.x]);
 
	__syncthreads();
 
	if(threadIdx.x % Mod == 0 && threadIdx.x != 0)
		atomicAdd(&rowSum[0], rowSum[threadIdx.x]);
 
	__syncthreads();
 
	if(threadIdx.x == 0)
		atomicAdd(sumHolder, rowSum[0]); 
}
__global__ void NormalizeWaveFunction2(ComplexNum const * const waveFunctionIn, ComplexNum * const waveFunctionOut, float const * const sumHolder)
//Same launch configuration as NormalizeWaveFunction.
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	float normalizationConst = sqrt(float( 1 / *sumHolder ));
	waveFunctionOut[index] = waveFunctionIn[index] * normalizationConst;
}
__global__ void PropogateWaveFunction(ComplexNum const * const waveFunctionIn, ComplexNum * const waveFunctionOut, float const * const devX4Coefficient, float const * const devX2Coefficient, float const * const devPotConst)
//Set waveFunctionOut to zero beforehand.
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ ComplexNum rowSum[THREADS_PER_BLOCK];
	float deltaX = (float) L/GRID_SIZE;
 
	ComplexNum K = KMatrix(blockIdx.y, index, devX4Coefficient, devX2Coefficient, devPotConst);
 
	ComplexNum kMatrixNormalization = 1. / ComplexNum(0, 2 * PI * TIME_STEP).SquareRoot();
 
	rowSum[threadIdx.x] = K * waveFunctionIn[index] * deltaX * kMatrixNormalization; 
 
	__syncthreads();
 
	int Mod = THREADS_PER_BLOCK / MOD_DIVISOR;
 
	if(threadIdx.x % Mod != 0)
	{
		atomicAdd(&rowSum[threadIdx.x - threadIdx.x % Mod].Re, rowSum[threadIdx.x].Re);
		atomicAdd(&rowSum[threadIdx.x - threadIdx.x % Mod].Im, rowSum[threadIdx.x].Im);
	}
 
	__syncthreads();
 
	if(threadIdx.x % Mod == 0 && threadIdx.x != 0)
	{
		atomicAdd(&rowSum[0].Re, rowSum[threadIdx.x].Re);
		atomicAdd(&rowSum[0].Im, rowSum[threadIdx.x].Im);
	}
 
	__syncthreads();
 
	if(threadIdx.x == 0)
	{
		atomicAdd(&waveFunctionOut[blockIdx.y].Re, rowSum[0].Re); 
		atomicAdd(&waveFunctionOut[blockIdx.y].Im, rowSum[0].Im); 
	}
 
	//if(index == 0)
	//	printf("P: %f\n", waveFunctionOut[blockDim.x * gridDim.x / 4]);
}
void ReshapeParent(int width, int height)
{
	parentWidth = width;
	parentHeight = height;
	glutSetWindow(parentWindow);
	glViewport(0,0,width,height);
	glutReshapeWindow(width, height);
}
void ReshapeMain(int width, int height)
{
	glutSetWindow(mainVisWindow);
	glutReshapeWindow(3 * parentWidth/4, parentHeight);
	glViewport(0,0,width,height);
	glutPositionWindow(0, 0);
}
void ReshapePos(int width, int height)
{
	glutSetWindow(avgPosWindow);
	glutReshapeWindow(parentWidth/4, parentHeight/3);
	glViewport(0,0,width,height);
	glutPositionWindow(3 * parentWidth/4, 0);
}
void ReshapeMomentum(int width, int height)
{
	glutSetWindow(avgMomentumWindow);
	glutReshapeWindow(parentWidth/4, parentHeight/3);
	glViewport(0,0,width,height);
	glutPositionWindow(3 * parentWidth/4, parentHeight/3);
}
void ReshapeEnergy(int width, int height)
{
	glutSetWindow(avgEnergyWindow);
	glutReshapeWindow(parentWidth/4, parentHeight/3);
	glViewport(0,0,width,height);
	glutPositionWindow(3 * parentWidth/4, 2 * parentHeight/3);
}
void Timer(int unused)
{
	char *windowLabels[4] = {"Wave Function", "Average Position", "Average Momentum", "Average Energy"};
	//int xRanges[][2] = {{-1,1},{0,1024},{0,1024},{0,1024}};
	//int yRanges[][2] = {{-.25,2},{-1.2,1.2},{-1.2,1.2},{3,4}};
	static unsigned int tailCounter;
	static unsigned int frames = 0;
 
	glutSetWindow(mainVisWindow);
	glColor3f(.3,.3,.3);
	if(tailCounter >= tailLength)
	{
		glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
		tailCounter = 0;
	}
	DrawString(-.95, 1.85, .001, .001, .001, GLUT_STROKE_ROMAN, windowLabels[0]);
	glColor3f(0,.6,0);
	DisplayPotential();
	glColor3f(0,0,1);
	DisplayMainVis();
	++tailCounter;
 
	glutSetWindow(timeWindow);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
	glColor3f(.3,.3,.3);
	DrawTime(-.9, .25, .0015, .005, .001, GLUT_STROKE_ROMAN, frames);
 
	glutSetWindow(avgPosWindow);
	glColor3f(.3,.3,.3);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
	DrawString(0, .6, .6, .002, .001, GLUT_STROKE_ROMAN, windowLabels[1]);
	DisplayAvgPos();
 
	glutSetWindow(avgMomentumWindow);
	glColor3f(.3,.3,.3);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
	DrawString(0, 1.2, .7, .002, .001, GLUT_STROKE_ROMAN, windowLabels[2]);
	DisplayAvgMomentum();
 
	glutSetWindow(avgEnergyWindow);
	glColor3f(.3,.3,.3);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
	DrawString(0, 6.5, .7, .007, .001, GLUT_STROKE_ROMAN, windowLabels[3]);
	DisplayAvgEnergy();
	DisplayAvgKEnergy();
 
	GLError(__LINE__);
 
	++frames;
 
	glutTimerFunc(pause, Timer, 0);
}
__device__ ComplexNum WaveFunction(const int index, float const * const devMinima)
{
	float alpha = 2;
	//Thin potential.
	//float x0 = -1;
	//For middling waves.
	//float x0 = -2;
 
	float x0 = -*devMinima;
 
	float x = index * (float)L/GRID_SIZE - (float) L/2;
 
	return ComplexNum(pow((float)(alpha/PI), (float).25) * glm::exp((float) -alpha * pow((float)x - x0, (float)2) / 2.),0);
}
 
 
CudaGLInterop::CudaGLInterop(const int width, const int height, const GLenum type)
	:WIDTH(width), HEIGHT(height), TYPE(type)
{
	glGenBuffers(1, &sharedBuf);
	GLError(__LINE__);
 
	glBindBuffer(TYPE, sharedBuf);
	GLError(__LINE__);
 
	switch(TYPE)
	{
		case GL_PIXEL_UNPACK_BUFFER:
			glBufferData(TYPE, WIDTH * HEIGHT * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW); 
			GLError(__LINE__);
			glActiveTexture(GL_TEXTURE0);
			glGenTextures(1, &sharedTex);
			glBindTexture(GL_TEXTURE_2D, sharedTex);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			GLError(__LINE__);
		break;
		case GL_ARRAY_BUFFER:
			glBufferData(TYPE, WIDTH * HEIGHT * sizeof(VertColor), NULL, GL_DYNAMIC_DRAW); 
			GLError(__LINE__);
		break;
	}
 
	cudaError_t cudaErr = cudaGraphicsGLRegisterBuffer(&resource, sharedBuf, cudaGraphicsMapFlagsNone);
	CudaError(__LINE__);
 
	glBindBuffer(TYPE, 0);
	GLError(__LINE__);
}
 
void * CudaGLInterop::MapResource(void)
{
	cudaGraphicsMapResources(1, &resource, NULL);
	CudaError(__LINE__);
 
	cudaGraphicsResourceGetMappedPointer((void**) &devPtr, &size, resource);
	CudaError(__LINE__);
 
	cudaMemset(devPtr, 0, size);
	CudaError(__LINE__);
 
	return devPtr;
}
 
void CudaGLInterop::UnmapResource(void)
{
	cudaGraphicsUnmapResources(1, &resource, NULL);
 
	CudaError(__LINE__);
}
 
void CudaGLInterop::Draw(GLenum drawMode, int end) const
{
	switch(TYPE)
	{
		case GL_PIXEL_UNPACK_BUFFER:
			glBindBuffer(TYPE, sharedBuf); 
			glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
			glBindBuffer(TYPE, 0);
			GLError(__LINE__);
		break;
		case GL_ARRAY_BUFFER:
			glEnable(GL_BLEND);
			glBindBuffer(TYPE, sharedBuf); 
 
			glEnableClientState(GL_COLOR_ARRAY);
			glEnableClientState(GL_VERTEX_ARRAY);
 
			unsigned char *offset = 0;
 
			glVertexPointer(3, GL_FLOAT, sizeof(VertColor), offset);
			glColorPointer(4, GL_UNSIGNED_BYTE, sizeof(VertColor), offset + 3 * sizeof(float));
 
			glDrawArrays(drawMode, 0, end);
 
			glDisableClientState(GL_COLOR_ARRAY);
			glDisableClientState(GL_VERTEX_ARRAY);
			glBindBuffer(TYPE, 0);
			glDisable(GL_BLEND);
			GLError(__LINE__);			
		break;
	}
}
 
void CudaGLInterop::DrawPixAsTex(void) const
{
	glEnable(GL_TEXTURE_2D);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, sharedBuf);
	glBindTexture(GL_TEXTURE_2D, sharedTex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
 
	glBegin(GL_QUADS);
		glTexCoord2f(0,1);
		glVertex3f(-1,-1,-.1);
 
		glTexCoord2f(0,.5);
		glVertex3f(-1,.001,-.1);
 
		glTexCoord2f(1,.5);
		glVertex3f(1,.001,-.1);
 
		glTexCoord2f(1,1);
		glVertex3f(1,-1,-.1);
	glEnd();
	glDisable(GL_TEXTURE_2D);
}
