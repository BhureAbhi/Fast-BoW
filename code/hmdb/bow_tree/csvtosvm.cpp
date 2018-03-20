#include <iostream>
#include <armadillo>
#include <string>
#include <ctime>
#include <sstream>
#include <vector>
using namespace std;
using namespace arma;


int main(int argc, char const *argv[]) {
    wall_clock tm;
    string X_file=argv[1];
    string Y_file=argv[2];
    string svm_file=argv[3];
    mat X,Y;
    X.load(X_file,auto_detect);
    Y.load(Y_file,auto_detect);

    ofstream outfile;
    outfile.open(svm_file);

    tm.tic();
    for(int i=0;i<(int)(X.n_rows);i++){
        outfile << Y[i];
        for(int j=0;j<(int)(X.n_cols);j++){
            if(X(i,j)!=0){
                outfile <<" "<<(j+1)<<":"<<X(i,j);
            }
        }
        outfile<<endl;
    }
    outfile.close();
    cout << "Fineshed. Time Taken: " << tm.toc() << endl;
    return 0;
}
