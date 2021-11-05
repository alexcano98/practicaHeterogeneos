#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace cv;

float * createFilter(int width)
{
        const float sigma       = 2.f;                          // Standard deviation of the Gaussian distribution.

        const int       half    = width / 2;
        float           sum             = 0.f;


        // Create convolution matrix
        float * res=(float *) malloc(width*width*sizeof(float)); //int to long?


        // Calculate filter sum first
        for (int r = -half; r <= half; ++r)
        {
                for (int c = -half; c <= half; ++c)
                {
                        // e (natural logarithm base) to the power x, where x is what's in the brackets
                        float weight = expf(-static_cast<float>(c * c + r * r) / (2.f * sigma * sigma));
                        int idx = (r + half) * width + c + half;

                        res[idx] = weight;
                        sum += weight;
                }
        }

        // Normalize weight: sum of weights must equal 1
        float normal = 1.f / sum;

        for (int r = -half; r <= half; ++r)
        {
                for (int c = -half; c <= half; ++c)
                {
                        int idx = (r + half) * width + c + half;

                        res[idx] *= normal;
                }
        }
        return res;
}


// Copmute gaussian blur per channel on the CPU.
// Call this function for each of red, green, and blue channels
// Returns blurred channel.
void ComputeConvolutionOnCPU(unsigned char* const blurredChannel, const unsigned char* const inputChannel, int rows, int cols, float * filter, int filterWidth)
{
        // Filter width should be odd as we are calculating average blur for a pixel plus some offset in all directions

        const int half   = filterWidth / 2;
        const int width  = cols - 1;
        const int height = rows - 1;

        // Compute blur
        for (int r = 0; r < rows; ++r)
        {
                for (int c = 0; c < cols; ++c)
                {
                        float blur = 0.f;

                        // Average pixel color summing up adjacent pixels.
                        for (int i = -half; i <= half; ++i)
                        {
                                for (int j = -half; j <= half; ++j)
                                {
                                        // Clamp filter to the image border
                                        int h = min(max(r + i, 0), height);
                                        int w = min(max(c + j, 0), width);

                                        // Blur is a product of current pixel value and weight of that pixel.
                                        // Remember that sum of all weights equals to 1, so we are averaging sum of all pixels by their weight.
                                        int             idx             = w + cols * h;                                                                                 // current pixel index
                                        float   pixel   = static_cast<float>(inputChannel[idx]);

                                        idx                             = (i + half) * filterWidth + j + half;
                                        float   weight  = filter[idx];

                                        blur += pixel * weight;
                                }
                        }

                        blurredChannel[c + cols * r] = static_cast<unsigned char>(blur);
                }
        }
}

void GaussianBlurOnCPU(uchar4* const modifiedImage, const uchar4* const rgba, int rows, int cols, float * filter, int  filterWidth)
{
        const int numPixels = rows * cols;

        // Create channel variables
        unsigned char* red                      = new unsigned char[numPixels];
        unsigned char* green            = new unsigned char[numPixels];
        unsigned char* blue                     = new unsigned char[numPixels];

        unsigned char* redBlurred       = new unsigned char[numPixels];
        unsigned char* greenBlurred = new unsigned char[numPixels];
        unsigned char* blueBlurred      = new unsigned char[numPixels];

        // Separate RGBAimage into red, green, and blue components
        for (int p = 0; p < numPixels; ++p)
        {
                uchar4 pixel = rgba[p];

                red[p]   = pixel.x;
                green[p] = pixel.y;
                blue [p] = pixel.z;
        }

        // Compute convolution for each individual channel



        ComputeConvolutionOnCPU(redBlurred, red, rows, cols, filter, filterWidth);
        ComputeConvolutionOnCPU(greenBlurred, green, rows, cols, filter, filterWidth);
        ComputeConvolutionOnCPU(blueBlurred, blue, rows, cols, filter, filterWidth);



        // Recombine channels back into an RGBAimage setting alpha to 255, or fully opaque
        for (int p = 0; p < numPixels; ++p)
        {
                unsigned char r = redBlurred[p];
                unsigned char g = greenBlurred[p];
                unsigned char b = blueBlurred[p];

                modifiedImage[p] = make_uchar4(r, g, b, 255);
        }

        delete[] red;
        delete[] green;
        delete[] blue;
        delete[] redBlurred;
        delete[] greenBlurred;
        delete[] blueBlurred;
}

__global__ void  ComputeConvolutionOnGPU(unsigned char* const blurredChannel, const unsigned char* const inputChannel, int rows, int cols, float* filter, int filterWidth)
{

    int c = blockIdx.x * blockDim.x + threadIdx.x; //row
    int r = blockIdx.y * blockDim.y + threadIdx.y; //colum


    if (rows > r && cols > c) { //si esta dentro...

        const int half = filterWidth / 2;
        const int width = cols - 1;
        const int height = rows - 1;
        
        float blur = 0.f;
        // Average pixel color summing up adjacent pixels.
        for (int i = -half; i <= half; ++i)
        {
            for (int j = -half; j <= half; ++j)
            {
                // Clamp filter to the image border
                int h = min(max(r + i, 0), height);
                int w = min(max(c + j, 0), width);

                // Blur is a product of current pixel value and weight of that pixel.
                // Remember that sum of all weights equals to 1, so we are averaging sum of all pixels by their weight.
                int             idx = w + cols * h;                                                                                 // current pixel index
                float   pixel = static_cast<float>(inputChannel[idx]);

                idx = (i + half) * filterWidth + j + half;
                float   weight = filter[idx];

                blur += pixel * weight;
            }
        }

        blurredChannel[c + cols * r] = static_cast<unsigned char>(blur);

    }


}

void GaussianBlurOnGPU(uchar4* const modifiedImage, const uchar4* const rgba, int rows, int cols, float* filter, int  filterWidth, 
        dim3 blockShape, int numBlocksFila, int numBlocksColumna, int blocksPorGrid)
{
    const int numPixels = rows * cols;

    // Create channel variables
    unsigned char* red = new unsigned char[numPixels];
    unsigned char* green = new unsigned char[numPixels];
    unsigned char* blue = new unsigned char[numPixels];

    unsigned char* redBlurred = new unsigned char[numPixels];
    unsigned char* greenBlurred = new unsigned char[numPixels];
    unsigned char* blueBlurred = new unsigned char[numPixels];


    // Separate RGBAimage into red, green, and blue components
    for (int p = 0; p < numPixels; ++p)
    {
        uchar4 pixel = rgba[p];

        red[p] = pixel.x;
        green[p] = pixel.y;
        blue[p] = pixel.z;
    }

    // Compute convolution for each individual channel

    unsigned char* dev_red;// = new unsigned char[numPixels];
    unsigned char* dev_redBlurred;// = new unsigned char[numPixels];

    cudaMalloc(&dev_red, numPixels * sizeof(char));
    cudaMalloc(&dev_redBlurred, numPixels * sizeof(char));

    cudaStream_t streams1[numBlocksColumna];
    cudaStream_t streams2[numBlocksColumna];
    cudaStream_t streams3[numBlocksColumna];

    /*size = N * sizeof(float) / nStreams;
    for (i = 0; i < nStreams; i++) {
        offset = i * N / nStreams;
        cudaMemcpyAsync(a_d + offset, a_h + offset, size, dir, stream[i]);
        kernel << <N / (nThreads * nStreams), nThreads, 0,
            stream[i] >> > (a_d + offset);
    }*/


    int gridSizePx = blockDim.y * cols * sizeof(unsigned char);
    int addRow = cols * sizeof(unsigned char);
    int ultima = numBlocksColumna - 1;

    for (int i = 0; i < numBlocksColumna-1; i++ ){
            cudaStreamCreate(&streams1[i]);

            if(i != ultima)
                cudaMemcpyAsync(dev_red + gridSizePx *i, red + gridSizePx *i, gridSizePx + addRow, cudaMemcpyHostToDevice,streams1[i]);
            else
                cudaMemcpyAsync(dev_red + gridSizePx * i, red + gridSizePx * i, gridSizePx, cudaMemcpyHostToDevice, streams1[i]);

            ComputeConvolutionOnGPU<<<numBlocksFila,blockDim,0,streams1[i]>>>(dev_redBlurred + i *gridSizePx, 
                    dev_red + i * gridSizePx, blockDim.x, cols, filter, filterWidth);
    }
    
  
    unsigned char* dev_green; //= new unsigned char[numPixels];
    unsigned char* dev_greenBlurred; //= new unsigned char[numPixels];

    cudaMalloc(&dev_green, numPixels * sizeof(char));
    cudaMalloc(&dev_greenBlurred, numPixels * sizeof(char));

    //ComputeConvolutionOnGPU(greenBlurred, green, rows, cols, filter, filterWidth);
    for (int i = 0; i < numBlocksColumna; i++) {
        cudaStreamCreate(&streams2[i]);

        if (i != ultima)
            cudaMemcpyAsync(dev_green + gridSizePx * i, green + gridSizePx * i, gridSizePx+ addRow, cudaMemcpyHostToDevice, streams2[i]);
        else
            cudaMemcpyAsync(dev_green + gridSizePx * i, green + gridSizePx * i, gridSizePx, cudaMemcpyHostToDevice, streams2[i]);

        ComputeConvolutionOnGPU << <numBlocksFila, blockDim,0,streams2[i] >> > (dev_greenBlurred + i * gridSizePx,
            dev_green + i * gridSizePx, blockDim.x, cols, filter, filterWidth);
    }


    unsigned char* dev_blue; //= new unsigned char[numPixels];
    unsigned char* dev_blueBlurred; // = new unsigned char[numPixels];

    cudaMalloc(&dev_blue, numPixels * sizeof(char));
    cudaMalloc(&dev_blueBlurred, numPixels * sizeof(char));


    //ComputeConvolutionOnGPU(blueBlurred, blue, rows, cols, filter, filterWidth);
    for (int i = 0; i < numBlocksColumna; i++) {
        cudaStreamCreate(&streams3[i]);
        if (i != ultima)
            cudaMemcpyAsync(dev_blue + gridSizePx * i, blue + gridSizePx * i, gridSizePx+ addRow, cudaMemcpyHostToDevice, streams3[i]);
        else
            cudaMemcpyAsync(dev_blue + gridSizePx * i, blue + gridSizePx * i, gridSizePx, cudaMemcpyHostToDevice, streams3[i]);
        ComputeConvolutionOnGPU << <numBlocksFila, blockDim,0, streams3[i] >> > (dev_blueBlurred + i * gridSizePx,
            dev_blue + i * gridSizePx, blockDim.x, cols, filter, filterWidth);
    }


    for (int i = 0; i < numBlocksColumna; i++) {
        cudaStreamSynchronize(streams1[i]);
        cudaStreamDestroy(streams1[i]);
    }

    //Se hace la primera copia de resultados
    

    for (int i = 0; i < numBlocksColumna; i++) {
        cudaStreamSynchronize(streams2[i]);
        cudaStreamDestroy(streams2[i]);
    }

    //Se hace la segunda copia de resultados


    for (int i = 0; i < numBlocksColumna; i++) {
        cudaStreamSynchronize(streams3[i]);
        cudaStreamDestroy(streams3[i]);
    }

    //Se hace la tercera copia de resultados


    // Recombine channels back into an RGBAimage setting alpha to 255, or fully opaque
    for (int p = 0; p < numPixels; ++p)
    {
        unsigned char r = redBlurred[p];
        unsigned char g = greenBlurred[p];
        unsigned char b = blueBlurred[p];

        modifiedImage[p] = make_uchar4(r, g, b, 255);
    }

    delete[] red;
    delete[] green;
    delete[] blue;
    delete[] redBlurred;
    delete[] greenBlurred;
    delete[] blueBlurred;
}



// Main entry into the application
int main(int argc, char** argv)
{
	char * imagePath;
	char * outputPath;
	
	int height, width, channels;
	uchar4 * originalImage, * blurredImage;

	int filterWidth=9;
	float * filter=createFilter(filterWidth);

	if (argc > 2)
	{
		imagePath = argv[1];
		outputPath = argv[2];
	}
	else
	{
		printf("Please provide input and output image files as arguments to this application.");
		exit(1);
	}


	Mat img;
	Mat imgRGBA;
	
	//Read the image
	img=imread(imagePath, IMREAD_COLOR);
	if(img.empty()) printf("Could not load image file: %s\n",imagePath);

	// get the image data
	height    = img.rows;
	width     = img.cols;
	channels =img.channels();

	cvtColor(img, imgRGBA, COLOR_BGR2BGRA);

	//Allocate and copy
	originalImage=(uchar4 *)imgRGBA.ptr<unsigned char>(0);;
	blurredImage=(uchar4 *)malloc(width*height*sizeof(uchar4));

	//Tu práctica empieza aquí
	//CUDA	

    dim3 blockShape = dim3(8, 8);
    int numBlocksFila = height/ 64 + 1;
    int numBlocksColumna = width / 64 + 1;
    int blocksPorGrid = numBlocksFila;
    
    GaussianBlurOnGPU(blurredImage, originalImage, height, width, filter, filterWidth, blockShape, 
            numBlocksFila, numBlocksColumna, blocksPorGrid);

	//Version CPU (Comentar cuando se trabaje con la GPU!)
	//GaussianBlurOnCPU(blurredImage, originalImage, height, width, filter, filterWidth);

	Mat out(height, width, CV_8UC4, blurredImage);

	imwrite(outputPath, out);

	printf("Done!\n");
	return 0;
}
