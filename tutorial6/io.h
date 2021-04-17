#ifndef IO_H
#define IO_H

#include <iostream>
#include <fstream>

using namespace std;

void inline write_to_file(double** A, int n, int rank){
	ofstream out("output/output.bin"+ to_string(rank), ios::out | ios::binary);
	if(!out) {
		cout << "Error: Cannot open file.\n";
		exit(1);
	}
	out.write((char *) A[0], sizeof(double)*n*n);
	out.close();
}

void inline verify(double** A, int n, int rank){
		double *correct = new double[n*n];
		ifstream in("output/output.bin" + to_string(rank), ios::in | ios::binary);
		if(!in) {
			//cout << "Cannot open output.bin. returning without verification.\n";
			return;
		}
		in.read((char *) correct, sizeof(double) * n*n);
		in.close();

		//compare
//#pragma omp parallel for
		for(int i=0; i<n; i++){
			for(int j=0; j<n; j++){
				if(fabs(A[i][j] - correct[i*n + j]) > 1e-9 ){
					printf("########## Output does not match. Error in %s %d: at cell (%d, %d) -> correct: %f \t computed:%f \t diff: %f\n"
							, __FILE__, __LINE__, i, j, correct[i*n + j], A[i][j], fabs(A[i][j] - correct[i*n + j]));
					exit(1);
				}
			}
		}

		printf("output matched for rank %d!!\n", rank);

		delete[] correct;
}

#endif	// EXPORT_H
