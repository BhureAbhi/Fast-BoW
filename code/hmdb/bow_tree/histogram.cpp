#include <iostream>
#include <armadillo>
#include <string>
#include <ctime>
#include <sstream>
#include <vector>
#include <cmath>
using namespace std;
using namespace arma;

// function  tree
struct node{
    int cluster_idx;
    mat c0;
    mat c1;
    node *left;
    node *right;
};

int aaa;
int idxcount=1;

int findcluster(mat x, node * root){
    int dim = (int)(x.n_cols);
    node * temp=root;
    //cout << "p1" << endl;
    while(true){
        if(temp->left==NULL && temp->right==NULL){
            return temp->cluster_idx;
        }
        else{
            double sum0 = 0;
            double sum1 =0;
            //cout << "p2" << endl;
            for(int d=0; d<dim; d++){
                sum0 += (x(0,d) - temp->c0(0,d))*(x(0,d) - temp->c0(0,d));
            }
            //cout << "p3" << endl;
            for(int d=0; d<dim; d++){
                sum1 += (x(0,d) - temp->c1(0,d))*(x(0,d) - temp->c1(0,d));
            }
            if(sum0<sum1){
                temp = temp->left;
            }
            else{
                temp = temp->right;
            }
        }
    }

    return -1;
}

node * buildtree(mat X, int height, int epochs){
    cout<<"sumit3"<<endl;
    int N = (int)(X.n_rows);
    int dim = (int)(X.n_cols);
    node *nd= new node;
    //cout << "X :"  << X.n_rows << " " << X.n_cols << endl;
    //cout<<"height ="<<" "<<height<<endl;
    if(height==0 || N<2){
        nd->cluster_idx = idxcount;
        nd->left = NULL;
        nd->right = NULL;
        idxcount++;
        return nd;
    }
    mat clusters;
    nd->cluster_idx = -1;
    //cout<< "kmeans starting"<<endl;
    cout<<"sumit4"<<endl;
    X=X.t();
    bool status = kmeans(clusters, X, 2, random_subset, epochs, false);
    X=X.t();
    cout<<"sumit5"<<endl;
    if(status==false){
        cout << "clustering failed." << endl;
        return NULL;
    }
    //cout << "clusters: " << clusters.n_rows << "  " << clusters.n_cols << endl;
    clusters=clusters.t();
    nd->c0=clusters.row(0);
    nd->c1=clusters.row(1);
    //cout << "clusters: " << clusters.n_rows << "  " << clusters.n_cols << endl;

    int ct0=0, ct1=0;
    vector<int> res(N);
    for(int i=0;i<N;i++){
        double sum0 = 0;
        double sum1 =0;
        // if(i%10000==0)
        //     cout << i << endl;
        for(int d=0; d<dim; d++){
            sum0 += (X(i,d) - (nd->c0)(0,d))*(X(i,d) - (nd->c0)(0,d));
        }
        for(int d=0; d<dim; d++){
            sum1 += (X(i,d) - (nd->c1)(0,d))*(X(i,d) - (nd->c1)(0,d));
        }
        if(sum0<sum1){
            res[i] = 0;
            ct0++;
        }
        else{
            res[i] = 1;
            ct1++;
        }
    }
    mat split0(ct0, dim);
    mat split1(ct1, dim);
    ct0=0;
    ct1=0;
    for(int i=0;i<N;i++){
        // if(i%10000==0)
        //     cout << i << endl;
        if(res[i]==0){
            split0.row(ct0) = X.row(i);
            ct0++;
        }
        else{
            split1.row(ct1) = X.row(i);
            ct1++;
        }
    }

    //cout << "split0: " << split0.n_rows << "  " << split0.n_cols << endl;
    //cout << "split1: " << split1.n_rows << "  " << split1.n_cols << endl << endl;

    nd->left = buildtree(split0, height-1, epochs);
    nd->right = buildtree(split1, height-1, epochs);

    return nd;
}

int main(int argc, char **argv){

    wall_clock tm;
    string train_file_path = argv[1];
    string X_file = argv[2];
    string train_hist_file = argv[3];
    int n_videos_train = atoi(argv[4]);
    int K = atoi(argv[5]);
    int epochs =atoi(argv[6]);
    string test_file_path = argv[7];
    string test_hist_file = argv[8];
    int n_videos_test = atoi(argv[9]);
      cout<<"sumit1"<<endl;
    mat X;
    X.load(X_file, auto_detect);
    int N = (int)(X.n_rows);
  cout<<"sumit2"<<endl;
    umat train_hist(n_videos_train, K);
    train_hist.zeros();

    //vector<mat> V(n_videos);

    // build tree from C.
    int height = (int)(log2(K));
    node *root= new node;
    tm.tic();
    root=buildtree(X, height, epochs);

    cout<<"(Model Build Time) Time Taken to build tree is"<<tm.toc()<<endl;


    double tt=0;
    int vct=0;
    int hct=0;
    while(vct!=n_videos_train){
        int ct=0;
        vector<mat> V(2500);
        while(ct!=2500 && vct!=n_videos_train){
            V[ct].load(train_file_path+"/"+to_string(vct+1)+".txt", auto_detect);
            //cout << vct << ":" << ct << "  ";
            ct++;
            vct++;
        }
        //cout<<ct<<" "<<endl;
        for(int i=0; i<2500; i++){
            if(hct==n_videos_train){
                break;
            }
            int Nv = (int)(V[i].n_rows);
            tm.tic();
            cout<<"V[i]="<<  Nv <<" * "<< V[i].n_cols<<endl;

            for(int n=0; n<Nv; n++){
                //<<n<<" ";
                // search tree and get idx.
                int clidx = findcluster(V[i].row(n), root);
                //cout<<"clidx"<<" "<<clidx<<" ";

                train_hist(hct,clidx-1)++;
            }
            tt+=tm.toc();

            //cout << hct << " ";
            hct++;
        }
        //cout << endl;
    }
    cout << "(Hist Gen Time) Time taken to generate features/histogram = " << tt << " seconds." << endl;
    train_hist.save(train_hist_file, csv_ascii);
    cout << "Train Hist size = " << train_hist.n_rows << " by " << train_hist.n_cols << endl;
    cout << "Finished" << endl;

    umat test_hist(n_videos_test, K);
    test_hist.zeros();

    tt=0;
    vct=0;
    hct=0;
    while(vct!=n_videos_test){
        int ct=0;
        vector<mat> V(2500);
        while(ct!=2500 && vct!=n_videos_test){
            V[ct].load(test_file_path+"/"+to_string(vct+1)+".txt", auto_detect);
            //cout << vct << ":" << ct << "  ";
            ct++;
            vct++;
        }

        for(int i=0; i<2500; i++){
            if(hct==n_videos_test){
                break;
            }
            int Nv = (int)(V[i].n_rows);
            tm.tic();
            for(int n=0; n<Nv; n++){
                //cout<<n<<endl;
                // search tree and get idx.
                int clidx = findcluster(V[i].row(n), root);
                //cout<<"clidx"<<" "<<clidx<<endl;

                test_hist(hct,clidx-1)++;
            }
            tt+=tm.toc();

            //cout << hct << " ";
            hct++;
        }
        //cout << endl;
    }
    cout << "(Hist Gen Time) Time taken to generate features/histogram = " << tt << " seconds." << endl;
    test_hist.save(test_hist_file, csv_ascii);
    cout << "Test Hist size = " << test_hist.n_rows << " by " << test_hist.n_cols << endl;
    cout << "Finished" << endl;
    return 0;
}
