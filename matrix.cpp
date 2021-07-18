
// MATRIX

//1>> Spiral traversal on a Matrix // TC o(rc) // o(rc)
vector<int> spirallyTraverse(vector<vector<int> > arr, int ro, int co) {
    vector <int> ans;
    int t=0;int l=0;
    int b=ro-1;
    int r=co-1;
    int d=0;
    while(t<=b && l<=r){
        if(d==0){
            for(int i=l;i<=r;i++)    ans.push_back(arr[t][i]);
            t++;
        }
        else if(d==1){
            for(int i=t;i<=b;i++)    ans.push_back(arr[i][r]);
            r--;
        }
        else if(d==2){
            for(int i=r;i>=l;i--)   ans.push_back(arr[b][i]);
            b--;
        }
        else if(d==3){
            for(int i=b;i>=t;i--)   ans.push_back(arr[i][l]);
            l++;
        }
        d=(d+1)%4;
    }
    return ans;
}

//2>> Search a 2D Matrix
// corner approach // Tc o(m+n)
bool searchMatrix(vector<vector>& matrix, int target){
   int m=matrix.size();
   if(m==0) return false;
   int n=matrix[0].size();
   int top=0; int down=m-1;
   int left=0; int right=n-1;
   int i; int flag=0;
   while(top<=down && left<=right){
     for(i=left;i<=right;i++){
        if(matrix[top][i]==target){
            flag=1;
            break;
           }
        }
     top+=1;
     if(flag==1) break;
  }
   if(flag==0) return false;
   else return true;
}

//Binary search // TC o(m+logn)
bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int m= matrix.size();
        int n= matrix[0].size();
        int lo = 0;
        int hi = m*n-1;
        while(lo<=hi){
            int mid =(lo + (hi-lo)/2);
            if(matrix[mid/n][mid%n] == target) return true;
            if(matrix[mid/n][mid%n] < target) lo=mid+1;
            else hi=mid-1;
        }
        return false;
    }

//3>> Find median in a row wise sorted matrix
// Naive approach
store in array and sort to find median

// TC O(32 * R * log(C)) // SC o(1)
int median(vector<vector<int>> &A, int M, int N){
    int k = ( M*N + 1 )/2 ;
    int a = INT_MAX ;
    int b = INT_MIN ;
    for( int i = 0 ; i < M ; i++ ){
        a = min( a , A[i][0] ) ;
        b = max( b , A[i][N-1] ) ;
    }
    while( a < b ){
       int m = ( a + b )/2 ;
       int cnt = 0 ;
       for( int i = 0 ; i < M ; i++ )
          cnt = cnt+(upper_bound(A[i].begin(),A[i].end(),m)-A[i].begin());
       if( cnt < k )a = m + 1 ;
       else b = m ;
    }
    return a;
}

//4>> Row with max 1s in binary matrix // TC o(mn)
int rowWithMax1s(vector<vector<int> > arr, int n, int m) {
    int max=0; int ro;
    int i;
    for(i=0; i<n; i++){
        int j = m-1-max;
        while(j>=0){
            if(arr[i][j]==0) break;
            else{
                max++;
                j--;
                ro=i;
            }
        }
    }
    return max!=0 ? ro:-1;
}

//5>> Print elements in sorted order using row-column wise sorted matrix
// TC o(nnlogn) // SC o(nn)
int main(){
        int N;
        cin >> N;
        vector<int> arr(N*N);
        for(int i = 0; i < N*N; i++)
            cin >> arr[i];
        sort(arr.begin(), arr.end());
        for(int i = 0; i < N*N; i++)
            cout << arr[i] << " ";
        cout << endl;
	return 0;
}

//6>> Given a binary matrix. Find the maximum area of a rectangle
// formed only of 1s in the given matrix. // TC o(mn) // SC o(m)
int histogramArea(int *arr, int n){
    stack<int>s;
    int max_area=0, area=0;
    int i=0;
    while(i<n){
        if(s.empty() or arr[s.top()]<=arr[i]){
            s.push(i);
            i++;
        }
        else{
            int top=s.top();
            s.pop();
            if(s.empty()){
                area=arr[top]*i;
            }
            else{
                area=arr[top]*(i-s.top()-1);
            }
            max_area=max(area,max_area);
        }
    }
    ///When array becomes empty, pop all the elements of stack:
    while(!s.empty()){
        int top=s.top();
        s.pop();
        area=arr[top]*(s.empty()?i:(i-s.top()-1));
        max_area=max(area,max_area);
    }
    return max_area;
}

int maxArea(int M[MAX][MAX], int n, int m) {
    int max_area = 0;
    int *arr = new int[m];
    for(int j=0;j<m;j++){
        arr[j] = M[0][j];
    }
    int curr_area = histogramArea(arr, m);
    max_area = max(curr_area, max_area);
    for(int i=1;i<n;i++){
        for(int j=0;j<m;j++){
            if(M[i][j]==0){
                arr[j] =0;
            }
            else{
                arr[j]+=M[i][j];
            }
        }
        curr_area = histogramArea(arr,m);
        max_area = max(curr_area, max_area);
    }
    return max_area;
}

//7>> Find a specific pair in matrix // Naive approach
search for all possible pair in matrix in TC o(nnnn)
// TC o(nn)
int findMaxValue(int mat[][N]) {
    //stores maximum value
    int maxValue = INT_MIN;
    // maxArr[i][j] stores max of elements in matrix
    // from (i, j) to (N-1, N-1)
    int maxArr[N][N];
    // last element of maxArr will be same's as of
    // the input matrix
    maxArr[N-1][N-1] = mat[N-1][N-1];
    // preprocess last row
    int maxv = mat[N-1][N-1];  // Initialize max
    for (int j = N - 2; j >= 0; j--) {
        if (mat[N-1][j] > maxv)
            maxv = mat[N - 1][j];
        maxArr[N-1][j] = maxv;
    }
    // preprocess last column
    maxv = mat[N - 1][N - 1];  // Initialize max
    for (int i = N - 2; i >= 0; i--) {
        if (mat[i][N - 1] > maxv)
            maxv = mat[i][N - 1];
        maxArr[i][N - 1] = maxv;
    }
    // preprocess rest of the matrix from bottom
    for (int i = N-2; i >= 0; i--) {
        for (int j = N-2; j >= 0; j--){
            // Update maxValue
            if (maxArr[i+1][j+1] - mat[i][j] > maxValue)
                maxValue = maxArr[i + 1][j+1]-mat[i][j];
            // set maxArr (i, j)
            maxArr[i][j] = max(mat[i][j],
                             max(maxArr[i][j+1],maxArr[i+1][j]));
        }
    }
    return maxValue;
}

//8>> Rotate matrix by 90 degrees  // TC o(nn)
void rotate90Clockwise(int a[N][N]){
    for (int i = 0; i < N / 2; i++) {
        for (int j = i; j < N - i - 1; j++) {
            int temp = a[i][j];
            a[i][j] = a[N - 1 - j][i];
            a[N - 1 - j][i] = a[N - 1 - i][N - 1 - j];
            a[N - 1 - i][N - 1 - j] = a[j][N - 1 - i];
            a[j][N - 1 - i] = temp;
        }
    }
}
// TC o(nn)
void rotate90Clockwise(int arr[N][N]){
    for (int j = 0; j < N; j++){
        for (int i = N - 1; i >= 0; i--)
            cout << arr[i][j] << " ";
        cout << '\n';
    }
}

//9>> Kth smallest element in a row-cpumn wise sorted matrix
//TC o(nn) // SC o(nn)
int kthSmallest(int mat[][MAX], int n, int k){
  vector<int> v;
for(int i=0;i<n;i++){
    for(int j=0;j<n;j++){
        v.push_back(mat[i][j]);
    }
}
sort(v.begin(),v.end());
return v[k-1];
}

// TC o(nlogn) // SC o(n)
#define dataset pair<int, pair<int,int> >
bool compare(const dataset &d1, const dataset &d2){
    return (d1.first>d2.first);
}
int kthSmallest(int mat[MAX][MAX], int n, int k){
    priority_queue<dataset, vector<dataset>, decltype(&compare)> pq(compare);
    for(int i=0; i<n; i++){
        pq.push({mat[0][i],{0,i}});
    }
    int ans;
    while(k--){
        dataset temp = pq.top();
        ans = temp.first;
        pq.pop();
        int col = temp.second.second;
        int row = temp.second.first;
        if(row != n-1){
            row = row + 1;
            pq.push({mat[row][col],{row,col}});
        }
    }
    return ans;
}

//10>> Common elements in all rows of a given matrix
// Naive approach // TC o(mnlogn)
sort and find;
// TC o(mn) // SC o(mn)
void printCommonElements(int mat[M][N]) {
    unordered_map<int, int> mp;
    for (int j = 0; j < N; j++)
        mp[mat[0][j]] = 1;
    for (int i = 1; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (mp[mat[i][j]] == i){
                mp[mat[i][j]] = i + 1;
                if (i==M-1)
                  cout << mat[i][j] << " ";
            }
        }
    }
}
