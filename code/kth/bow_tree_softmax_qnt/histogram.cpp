#include <iostream>
#include <armadillo>
#include <string>
#include <ctime>
#include <sstream>
#include <vector>
#include <cmath>
#include <queue>
using namespace std;
using namespace arma;

// function  tree
struct node{
    int cluster_idx;
    vector<int> c0;
    vector<int> c1;
    node *left;
    node *right;
};

int aaa;


int findcluster(mat x, node * root, mat H){
    int dim = (int)(x.n_cols);
    node * temp=root;
    int U = H.n_rows;
    //cout<<"H"<<U<<endl;
    while(true){
        if(temp->left==NULL && temp->right==NULL){
            //cout<<"true"<<endl;
            return temp->cluster_idx;
        }
        else{
            float sum0 = 0;
            float sum1 =0;
            //float S[U]={0};
            float S[U];
            for(int u=0;u<U;u++)
			{
				S[u]=0;
			}
            //cout<<S[0]<<" "<<"sumit1"<<endl;
            for(int d=0; d<dim; d++){
                S[temp->c0[d]] += x(0,d);
            }
            S[temp->c0[dim]]+=1;
			for(int u=0;u<U;u++)
			{
				sum0+=S[u]*H(u);
			}

            for(int u=0;u<U;u++)
			{
				S[u]=0;
			}

            for(int d=0; d<dim; d++){
                S[temp->c1[d]] += x(0,d);
            }
            S[temp->c1[dim]]+=1;
			for(int u=0;u<U;u++)
			{
				sum1+=S[u]*H(u);
			}
            //cout<< "sumit2"<<endl;
            if(sum0>sum1){
                temp = temp->left;
            }
            else{
                temp = temp->right;
            }
        }
    }

    return -1;
}

int main(int argc, char **argv){

    wall_clock tm;
    string file_path = argv[1];
    string model_file = argv[2];
    string hist_file = argv[3];
    int n_videos = atoi(argv[4]);
    int K = atoi(argv[5]);
    int epochs =atoi(argv[6]);
    string hash_file = argv[7];

    mat X;
    X.load(model_file, auto_detect);
    int N = (int)(X.n_rows);

    mat H;
    H.load(hash_file, auto_detect);

    umat hist(n_videos, K);
    hist.zeros();

/////////////////////////////////////////////////////
    tm.tic();
    queue<node *> nod_queue;
    node *root = new node;
    nod_queue.push(root);
    int ndCt = 1;
    int idxcount=1;

    while(!nod_queue.empty()){
        node *curNode = nod_queue.front();
        nod_queue.pop();
        if(ndCt>=K){
            curNode->cluster_idx = idxcount;
            curNode->left = NULL;
            curNode->right = NULL;
            //cout << "leaf node " << ndCt << " index " << idxcount << endl;
            idxcount++;
            ndCt++;
            continue;
        }
        curNode->c0.resize(163);
        curNode->c1.resize(163);
        for(int i=0; i<163; i++){
            curNode->c0[i] = X(2*ndCt-2,i);
        }
        for(int i=0; i<163; i++){
            curNode->c1[i] = X(2*ndCt-1,i);
        }
        // curNode->c0 = X.row(2*ndCt-2);
        // curNode->c1 = X.row(2*ndCt-1);
        node *lnode = new node;
        node *rnode = new node;
        curNode->left = lnode;
        curNode->right = rnode;
        nod_queue.push(lnode);
        nod_queue.push(rnode);
        curNode->cluster_idx = -1;
        //cout << "in node " << ndCt << endl;
        ndCt++;
    }

    cout << "(Model Build Time) Time taken to build tree = " << tm.toc() << " seconds." << endl;
//////////////////////////////////////////////////
    mat V;
    double tt=0;
    for(int v=1; v<=n_videos; v++){
        V.load(file_path+"/"+to_string(v)+".txt", auto_detect);
        //cout<<v<<endl;
        tm.tic();
        int Nv = (int)(V.n_rows);
        //cout<< V.n_cols<<endl;
        for(int n=0; n<Nv; n++){
            //cout<<n<<endl;
            // search tree and get idx.
            int clidx = findcluster(V.row(n), root, H);
            hist(v-1,clidx-1)++;
        }
        tt += tm.toc();
    }
    cout << "(Hist Gen Time)Time taken to generate features/histogram = " << tt << " seconds." << endl;
    hist.save(hist_file, csv_ascii);
    cout << "Hist size = " << hist.n_rows << " by " << hist.n_cols << endl;
    cout << "Finished" << endl;

    return 0;
}
