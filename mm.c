/* CPSC 479
 * Topic: Matrix Multiplication
 * 
 * Notes and Assumptions:
 * This program assumes that the txt files provided has the correct number
 * of rows and columns for the matrices
 * 
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

//Allocate size of the matrix
int mallocMatrix(int ***matrix, int row, int col) {
	int *m = (int*)malloc(row*col*sizeof(int));
	if(!m) {
		 return -1;
	}
	(*matrix) = (int**)malloc(row*sizeof(int*));
	if(!(*matrix)) {
		printf("freeing m\n");
		free(m);
		printf("freed m\n");
		return -1;
	}
	
	//Set up the pointers into contigious memory
	for(int i = 0; i < row; i++) {
		(*matrix)[i] = &(m[i*col]);
	}
	
	return 0;
}

//Free 2D matrix memory
int freeMatrix(int ***matrix) {
	free(&((*matrix)[0][0]));
	free(*matrix);
	return 0;
}

//Function to read from matrix.txt
void readMatrix(FILE *file, int **matrix, int row, int col) {
	for(int i = 0; i < row; i++) {
		for(int j = 0; j < col; j++) {
			fscanf(file, "%d", &(matrix[i][j]));
		}
	}
}

//Function to print out the matrix
void printMatrix(int **matrix, int row, int col) {
	for(int i = 0; i < row; i ++) {
		for(int j = 0; j < col; j++) {
			printf("%d ", matrix[i][j]);
		}
		printf("\n");
	}
}

int main(int argc, char **argv) {
	int **matrixA, **matrixB;
	int rank, size, rowA, rowB, colA, colB;
	FILE *file;
	MPI_Status status;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	//Read in the matrix in process 0
	if(rank == 0) {
		//Read in matrixA
		file = fopen("matrixA.txt", "r");
		if(file == NULL) {
			printf("matrixA.txt could not be opened.\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		
		//Get the values on the first row of matrixA.txt
		fscanf(file, "%d %d", &rowA, &colA);
		mallocMatrix(&matrixA, rowA, colA);
		readMatrix(file, matrixA, rowA, colA);
		fclose(file);
		
		//Read in matrixB
		file = fopen("matrixB.txt", "r");
		if(file == NULL) {
			printf("matrixB.txt could not be opened.\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		
		//Get the values from the first row of matrixB.txt
		fscanf(file, "%d %d", &rowB, &colB);
		mallocMatrix(&matrixB, rowB, colB);
		readMatrix(file, matrixB, rowB, colB);
		fclose(file);
		
		//Check if matrixA's # of columns is equal to matrixB's # of rows
		if(colA != rowB) {
			printf("Error!\nMatrixA's number of columns has to equal MatrixB's number of rows.\n");
			MPI_Abort(MPI_COMM_WORLD, 2);
		}
		
		printMatrix(matrixA, rowA, colA);
		printf("\n\n");
		printMatrix(matrixB, rowB, colB);
	}
	
	//Ending the program
	if(rank == 0) {
		freeMatrix(&matrixA);
		freeMatrix(&matrixB);
	}
	MPI_Finalize();
	return 0;
}
