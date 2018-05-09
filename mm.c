/* CPSC 479
 * Members: Johnson Lien, Sean McKean
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
		//Check number of processes is equal to the number of rows of the first matrix
		if(size != rowA) {
			printf("Error!\nThe number of processes must equal the number of rows in Matrix A!\n");
			printf("Number of processes expected: %d\n", rowA);
			MPI_Abort(MPI_COMM_WORLD, 3);
		}
		
		printf("Matrix A:\n");
		printMatrix(matrixA, rowA, colA);
		printf("\nMatrixB:\n");
		printMatrix(matrixB, rowB, colB);
		
	} 
	
	//Broadcast our rows and columns to all other processes
		MPI_Bcast(&rowA, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&rowB, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&colA, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&colB, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	if(rank != 0) {
		//printf("colB: %d\nrowB: %d\n", colB, rowB);
		mallocMatrix(&matrixB, rowB, colB);
	}
	
	//Broadcast matrix B by sending out rows 
	for(int i = 0; i < rowB; i++) {
		MPI_Bcast(&matrixB[i][0], colB, MPI_INT, 0, MPI_COMM_WORLD);
	}
	
	//print out rank 1's matrix B
	if(rank == 1) {
		printf("rank %d\n", rank);
		printMatrix(matrixB, rowB, colB);
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	//All ranks, at this point, all have matrix B
	
	int *matrixARow = (int*)malloc(colA*sizeof(int)); 
	
	if(rank == 0) {
		for(int i = 0; i < size; i++) {
			for(int j = 0; j < colA; j++) {
				matrixARow[j] = matrixA[i][j];
				MPI_Send(&matrixARow[j], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			}
		}
		
		//Rewrite our row 0 into matrixARow array since it got overwritten
		for(int i = 0; i < colA; i++) {
			matrixARow[i] = matrixA[0][i];
		}	
	}
	else {
		for(int i = 0; i < 1; i++) {
			for(int j = 0; j < colA; j++) {
				MPI_Recv(&matrixARow[j], 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
			}
		}
	}
	
	//Each section will do their part of the matrix multiplication
	int *matrixARowProduct = (int*)malloc(colB*sizeof(int));
	for(int i = 0; i < colB; i++) {
		int sum = 0;
		for(int j = 0; j < rowB; j++) {
			sum += matrixARow[j] * matrixB[j][i];
		}
		matrixARowProduct[i] = sum;
	}
	
	//Allocate memory for the solution matrix
	int **matrixSolution; 			//Resulting matrix will be of size rowA x colB
	mallocMatrix(&matrixSolution, rowA, colB);
	
	//Each process will save its portion calculations into their own version of the matrixSolution
	// Since each process computes its own row, we will save the results the row=rank matrixSolution
	for( int i = 0; i < colB; i++) {
		matrixSolution[rank][i] = matrixARowProduct[i];
	}
	
	//Send out each version of matrixSolution to rank 0
	if(rank != 0) {
		for(int i = 0; i < colB; i++) {
			printf("Rank %d sending out %d\n", rank, matrixSolution[rank][i]);
			MPI_Send(&matrixSolution[rank][i], 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		}

	}
	
	//Rank 0 will now receive from all processes
	else {
		for(int i = 1; i < rowA; i++) {
			for(int j = 0; j < colB; j++) {
				printf("Receiving from rank %d\n", i);
				MPI_Recv(&matrixSolution[i][j], 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
			}
		}
	}
	
	//Wait for all processes to finish calculations
	MPI_Barrier(MPI_COMM_WORLD);
	
	//Ending the program
	//Free allocated memory
	if(rank == 0) {
		printf("Matrix solution:\n");
		printMatrix(matrixSolution, rowA, colB);
		
		//Free memory
		freeMatrix(&matrixA);
		freeMatrix(&matrixB);
		freeMatrix(&matrixSolution);
		
		free(matrixARow);
		free(matrixARowProduct);
	}
	 
	MPI_Finalize();
	return 0;
}
