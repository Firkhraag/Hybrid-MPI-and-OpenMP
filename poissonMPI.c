#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include <mpi.h>

// Bool type
typedef enum {false, true} bool;

// Given functions
float u(float x, float y) {
    return exp(1 - (x + y) * (x + y));
}

float k(float x) {
    return 4 + x;
}

float q(float x, float y) {
    return (x + y) * (x + y);
}

float F(float x, float y) {
    return u(x, y) * ((x + y) * (x + y + 2) - (4 + x) * (-4 + 8 * (x + y) * (x + y)));
}

// Dot product
float dotProduct(float* grid1, float* grid2, int blockWidth, int blockHeight, float stepX, float stepY) {
    float result = 0;
    for (int i = 1; i < blockHeight - 1; i++) {
        for (int j = 1; j < blockWidth - 1; j++) {
            const int index = i * blockWidth + j;
            result += grid1[index] * grid2[index];
        }
    }
    result = result * stepX * stepY;

    float sum;
    // Gathers to root and reduce with sum: send_data, recv_data, count, datatype, op, root, communicator
    MPI_Reduce(&result, &sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    // Broadcasts from root to other processes: buffer, count, datatype, root, communicator
    MPI_Bcast(&sum, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    return sum;
}

// For each block
void passInformationBetweenProcesses(const int currentRank, const int numOfBlocksX, const int numOfBlocksY, const int blockPositionX, const int blockPositionY, float* grid,
                                    const int blockWidth, const int blockHeight) {

    bool up = true;
    if (blockPositionX == 0) {
        up = false;
    }
    bool bottom = true;
    if (blockPositionX == numOfBlocksX - 1) {
        bottom = false;
    }
    bool left = true;
    if (blockPositionY == 0) {
        left = false;
    }
    bool right = true;
    if (blockPositionX == numOfBlocksY - 1) {
        right = false;
    }

    float* sendUp;
    float* sendBottom;
    float* sendLeft;
    float* sendRight;

    /*
        Process grid
        - - - - - - - -
        - * * * * * * -
        - * + + + + * -
        - * + + + + * -
        - * + + + + * -
        - * + + + + * -
        - * * * * * * -
        - - - - - - - -

        - points that we receive from other processes
        * points that we send to other processes
        + internal points
    */
    const int width  = blockWidth - 2;
    const int height = blockHeight - 2;

    const int upperNeighborRank = numOfBlocksY * (blockPositionX - 1) + blockPositionY;
    const int bottomNeighborRank = numOfBlocksY * (blockPositionX + 1) + blockPositionY;
    const int leftNeighborRank = numOfBlocksY * blockPositionX + (blockPositionY - 1);
    const int rightNeighborRank = numOfBlocksY * blockPositionX + (blockPositionY + 1);

    if (currentRank == 1) {
        printf("upR: %d\n", upperNeighborRank);
        printf("bottomR: %d\n", bottomNeighborRank);
        printf("up: %d\n", up);
        printf("bottom: %d\n", bottom);
    }

    MPI_Status status;
    MPI_Request leftSendRequest, rightSendRequest, upSendRequest, bottomSendRequest;

    int i;
    // Sending nodes near boundary to other processes
    if (up == true) {
        sendUp = (float*)malloc(width * sizeof(float));
        for (i = 0; i < width; i++) {
            sendUp[i] = grid[blockWidth + (i + 1)];
        }
        // Nonblocking send: buf, count, datatype, destination, tag, communicator, request
        MPI_Isend(sendUp, width, MPI_FLOAT, upperNeighborRank, 0, MPI_COMM_WORLD, &upSendRequest);
	}
    if (bottom == true) {
        if (currentRank == 1) {
            printf("nooooo waaaay\n");
        }
        sendBottom = (float*)malloc(width * sizeof(float));
        for (i = 0; i < width; i++) {
            sendBottom[i] = grid[height * blockWidth + (i + 1)];
        }
        MPI_Isend(sendBottom, width, MPI_FLOAT, bottomNeighborRank, 0, MPI_COMM_WORLD, &bottomSendRequest);
	}
    if (left == true) {
        sendLeft = (float*)malloc(height * sizeof(float));
        for (i = 0; i < height; i++) {
            sendLeft[i] = grid[(i + 1) * blockWidth + 1];
        }
        MPI_Isend(sendLeft, height, MPI_FLOAT, leftNeighborRank, 0, MPI_COMM_WORLD, &leftSendRequest);
	}
    if (right == true) {
        sendRight = (float*)malloc(height * sizeof(float));
        for (i = 0; i < height; i++) {
            sendRight[i] = grid[(i + 1) * blockWidth + width];
        }
        MPI_Isend(sendRight, height, MPI_FLOAT, rightNeighborRank, 0, MPI_COMM_WORLD, &rightSendRequest);
	}

    // Receive boundary nodes from other processes
    if (up == true) {
        float* receiveUp = (float*)malloc(width * sizeof(float));
        // Blocking receive: buf, count, datatype, source, tag, communicator, status
        MPI_Recv(receiveUp, width, MPI_FLOAT, upperNeighborRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (i = 0; i < width; i++) {
            grid[i + 1] = receiveUp[i];
        }
        free(receiveUp);
    }
    if (bottom == true) {
        float* receiveBottom = (float*)malloc(width * sizeof(float));
        MPI_Recv(receiveBottom, width, MPI_FLOAT, bottomNeighborRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (i = 0; i < width; i++) {
            grid[(height + 1) * blockWidth + (i + 1)] = receiveBottom[i];
        }
        free(receiveBottom);
    }
    if (left == true) {
        float* receiveLeft = (float*)malloc(height * sizeof(float));
        MPI_Recv(receiveLeft, height, MPI_FLOAT, leftNeighborRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (i = 0; i < height; i++) {
            grid[(i + 1) * blockWidth] = receiveLeft[i];
        }
        free(receiveLeft);
    }
    if (right == true) {
        float* receiveRight = (float*)malloc(height * sizeof(float));
        MPI_Recv(receiveRight, height, MPI_FLOAT, rightNeighborRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (i = 0; i < height; i++) {
            grid[(i + 1) * blockWidth + (width + 1)] = receiveRight[i];
        }
        free(receiveRight);
    }

    // Wait sending to compelete and delete allocated arrays
    if (up == true) {
        // Wait for request completion
        MPI_Wait(&upSendRequest, &status);
        free(sendUp);
    }
    if (bottom == true) {
        MPI_Wait(&bottomSendRequest, &status);
        free(sendBottom);
    }
    if (left == true) {
        MPI_Wait(&leftSendRequest, &status);
        free(sendLeft);
    }
    if (right == true) {
        MPI_Wait(&rightSendRequest, &status);
        free(sendRight);
    }
}


int main(int argc, char **argv) {

    struct timeval start, end;

    // Get start time
    gettimeofday(&start, NULL);

    // Rectangle
    const float a1 = -1.0;
    const float a2 = 2.0;
    const float b1 = -2.0;
    const float b2 = 2.0;

    // Epsilon
    const float eps = 1e-5;

    // Square grid
    const int n = 20;

    // Step
    const float stepX = (a2 - a1) / n;
    const float stepY = (b2 - b1) / n;

    const float stepXCoeff = 1 / (stepX * stepX);
    const float stepYCoeff = 1 / (stepY * stepY);

    float tau;

	int size;
	int currentRank;

	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &currentRank);

    int numOfBlocksY = 1;
	while (size > 2 * numOfBlocksY * numOfBlocksY) {
		numOfBlocksY *= 2;
	}
	const int numOfBlocksX = size / numOfBlocksY;

	// Global block position
    const int blockPositionX = currentRank / numOfBlocksY;
    const int blockPositionY = currentRank % numOfBlocksY;

    const int blockSizeX = (n + 1) / numOfBlocksX;
	const int blockSizeY = (n + 1) / numOfBlocksY;

	const int startX = fmax(0, blockSizeX * blockPositionX);
	const int endX = blockPositionX + 1 < numOfBlocksX ? startX + blockSizeX : n;

	const int startY = fmax(0, blockSizeY * blockPositionY);
	const int endY = blockPositionY + 1 < numOfBlocksY ? startY + blockSizeY : n;

	const int blockHeight = endX - startX + 1;
	const int blockWidth = endY - startY + 1;

    // printf("------\n");
    // printf("Size: %d\n", size);
    // printf("Rank: %d\n", currentRank);
    // printf("NumOfBlocksY: %d\n", numOfBlocksY);
    // printf("NumOfBlocksX: %d\n", numOfBlocksX);
    // printf("BlockPositionX: %d\n", blockPositionX);
    // printf("BlockPositionY: %d\n", blockPositionY);
    // printf("BlockSizeX: %d\n", blockSizeX);
    // printf("BlockSizeY: %d\n", blockSizeY);
    // printf("StartX: %d\n", startX);
    // printf("EndX: %d\n", endX);
    // printf("StartY: %d\n", startY);
    // printf("EndY: %d\n", endY);
    // printf("BlockHeight: %d\n", blockHeight);
    // printf("BlockWidth: %d\n", blockWidth);
    // printf("------\n");

    // Local grid approximation array
    float* grid = (float*)malloc(blockWidth * blockHeight * sizeof(float));

    // Difference between two steps array
    float* gridDiff = (float*)malloc(blockWidth * blockHeight * sizeof(float));

    // Local real values array
    float* realValues = (float*)malloc(blockWidth * blockHeight * sizeof(float));

    // Local residuals array
    float* rk = (float*)malloc(blockWidth * blockHeight * sizeof(float));
    // Fill boundary with zeros
    for (int j = 0; j < blockWidth; j++) {
		rk[j] = 0;
    }
    for (int j = 0; j < blockWidth; j++) {
		rk[(blockHeight - 1) * blockWidth + j] = 0;
    }
    for (int i = 0; i < blockHeight; i++) {
		rk[i * blockWidth] = 0;
    }
    for (int i = 0; i < blockHeight; i++) {
		rk[i * blockWidth + (blockWidth - 1)] = 0;
    }

    // A * rk array
    float* ark = (float*)malloc(blockWidth * blockHeight * sizeof(float));

    float stopCondition;

    // Find real values in the block
	for (int i = 1; i < blockHeight - 1; i++) {
		for (int j = 1; j < blockWidth - 1; j++) {
			realValues[i * blockWidth + j] = u(a1 + (i + startX) * stepX, b1 + (j + startY) * stepY);
		}
	}

    // Find global boundary values in the block
	if (startX == 0) {
		for (int j = 0; j < blockWidth; j++) {
			grid[j] = u(a1 + startX * stepX, b1 + (j + startY) * stepY);
		}
	}

	if (endX == n) {
		for (int j = 0; j < blockWidth; j++) {
            grid[(blockHeight - 1) * blockWidth + j] = u(a1 + (blockHeight - 1 + startX) * stepX, b1 + (j + startY) * stepY);
		}
	}

	if (startY == 0) {
		for (int i = 0; i < blockHeight; i++) {
            grid[i * blockWidth] = u(a1 + (i + startX) * stepX, b1 + startY * stepY);
		}
	}

	if (endY == n) {
		for (int i = 0; i < blockHeight; i++) {
            grid[i * blockWidth + (blockWidth - 1)] = u(a1 + (i + startX) * stepX, b1 + (blockWidth - 1 + startY) * stepY);
		}
    }

    // Initializing grid with starting values
    for (int i = 1; i < blockHeight - 1; i++) {
		for (int j = 1; j < blockWidth - 1; j++) {
			grid[i * blockWidth + j] = 0;
		}
	}

    FILE *f = fopen("resultMPI.txt", "w");
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }
    int step = -1;
    float error;

    do {
        step++;
        // Find residual using difference scheme
        for (int i = 1; i < blockHeight - 1; i++) {
            for (int j = 1; j < blockWidth - 1; j++) {
                const float x = a1 + (i + startX) * stepX;
                const float y = b1 + (j + startY) * stepY;
                const int index = i * blockWidth + j;
                rk[index] = -(
                    stepXCoeff * (k(x + 0.5 * stepX) * (grid[index + blockWidth] - grid[index]) -
                    k(x - 0.5 * stepX) * (grid[index] - grid[index - blockWidth])) +
                    stepYCoeff * k(x) * ((grid[index + 1] - grid[index]) -
                    (grid[index] - grid[index - 1]))) +
                    q(x, y) * grid[index] - F(x, y);
                // printf("Value: %f\n", rk[index]);
            }
        }

        // Pass residuals to adjacent processes
        passInformationBetweenProcesses(currentRank, numOfBlocksX, numOfBlocksY, blockPositionX, blockPositionY, rk, blockWidth, blockHeight);

        // Find A * rk using difference scheme
        for (int i = 1; i < blockHeight - 1; i++) {
            for (int j = 1; j < blockWidth - 1; j++) {
                const float x = a1 + (i + startX) * stepX;
                const float y = b1 + (j + startY) * stepY;
                const int index = i * blockWidth + j;
                ark[index] = -(
                    stepXCoeff * (k(x + 0.5 * stepX) * (rk[index + blockWidth] - rk[index]) -
                    k(x - 0.5 * stepX) * (rk[index] - rk[index - blockWidth])) +
                    stepYCoeff * k(x) * ((rk[index + 1] - rk[index]) -
                    (rk[index] - rk[index - 1]))) +
                    q(x, y) * rk[index];
            }
        }

        // Find tau
        float tau1 = dotProduct(ark, rk, blockWidth, blockHeight, stepX, stepY);
        float tau2 = dotProduct(ark, ark, blockWidth, blockHeight, stepX, stepY);

        tau = tau1 / tau2;

        // Find new approximation
        for (int i = 1; i < blockHeight - 1; i++) {
            for (int j = 1; j < blockWidth - 1; j++) {
                const int index = i * blockWidth + j;
                const float gridElementOld = grid[index];
                grid[index] -= tau * rk[index];
                gridDiff[index] = grid[index] - gridElementOld;
            }
        }

        // Deviation
        error = 0;
        for (int i = 1; i < blockHeight - 1; i++) {
            for (int j = 1; j < blockWidth - 1; j++) {
                const int index = i * blockWidth + j;
                error += (realValues[index] - grid[index]) * (realValues[index] - grid[index]);
            }
        }

        float globalError = 0;
        // Gathers to root and reduce with sum: send_data, recv_data, count, datatype, op, root, communicator
        MPI_Reduce(&error, &globalError, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        if (currentRank == 0) {
            fprintf(f, "Step: %d. Error: %f\n", step, globalError);
        }

        stopCondition = sqrt(dotProduct(gridDiff, gridDiff, blockWidth, blockHeight, stepX, stepY));

        // Wait for all processes to complete the step
        MPI_Barrier(MPI_COMM_WORLD);

        // break;

    } while (stopCondition > eps);

    free(gridDiff);
    free(rk);
    free(ark);

    // End time
    gettimeofday(&end, NULL);
    // In seconds
    double time_taken = end.tv_sec + end.tv_usec / 1e6 -
                        start.tv_sec - start.tv_usec / 1e6;
    fprintf(f, "Execution time: %f\n", time_taken);

    // for (int i = 1; i < blockHeight - 1; i++) {
    //     for (int j = 1; j < blockWidth - 1; j++) {
    //         fprintf(f, "Original function: %f, Approximated result: %f\n", realValues[i * blockWidth + j], grid[i * blockWidth + j]);
    //     }
    // }

    fclose(f);
    free(realValues);
    free(grid);

    MPI_Finalize();
    return 0;
}