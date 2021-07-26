// DP

//1>> Coin ChangeProblem // TC o(mn) // SC o(mn)
long long int dp[1001][1001];
  long long int solve(int arr[], int m, int n){
      if(n>0 && m==-1) return 0;
      if(n==0) return 1;
      if(n<0) return 0;
      if(dp[m][n]!=-1) return dp[m][n];
      return dp[m][n] = solve(arr,m,n-arr[m]) + solve(arr,m-1,n);
  }
    long long int count( int S[], int m, int n ){
        dp[m][n+1];
        memset(dp,-1,sizeof(dp));
        return solve(S,m-1,n);
    }

//2>> Knapsack Problem // TC o(nw) // SC o(nw)
int dp[1001][1001];
int solve(int wt[], int val[], int w, int n){
    if(w==0 or n==-1) return 0;
    if(dp[n][w]!=-1) return dp[n][w];
    if(wt[n]>w) return dp[n][w] = solve(wt,val,w,n-1);
    int a,b;
    a = val[n] + solve(wt,val,w-wt[n],n-1);
    b = solve(wt,val,w,n-1);
    return dp[n][w] = max(a,b);
}
int knapSack(int w, int wt[], int val[], int n){
   dp[n][w];
   memset(dp,-1,sizeof(dp));
   return solve(wt,val,w,n-1);
}

//3>> Binomial CoefficientProblem // TC o(nr) // SC o(r)
int nCr(int n, int r){
       if(n<r) return 0;
       if((n-r)<r) r=n-r;
       int mod = 1000000007;
       int dp[r+1];
       memset(dp,0,sizeof(dp));
       dp[0]=1;
       for(int i=0; i<=n; i++){
           for(int j =min(r,i);j>0;j--){
               dp[j] = (dp[j]+dp[j-1])%mod;
           }
       }
       return dp[r];
    }

  //4>> Permutation CoefficientProblem // TC o(n)
  int PermutationCoeff(int n, int k) {
    int P = 1;
    for (int i = 0; i < k; i++)
        P *= (n-i) ;
    return P;
}

//5>> Edit Distance // TC o(nn) // SC o(nn)
int dp[1001][1001];
	int fun(string s, string t, int m, int n){
	    if(m==-1) return n+1;
	    if(n==-1) return m+1;

	    if(dp[m][n]!=-1) return dp[m][n];
	    if(s[m]==t[n]) return dp[m][n] = fun(s,t,m-1,n-1);

	    int a = fun(s,t,m-1,n-1);
	    int b = fun(s,t,m,n-1);
	    int c = fun(s,t,m-1,n);

	    return dp[m][n]=1+min(a,min(b,c));
	}
		int editDistance(string s, string t){
		    memset(dp,-1,sizeof(dp));
		    int m = s.length();
		    int n = t.length();
		    return fun(s,t,m-1,n-1);
		}

  //6>> Partition Equal Subset Sum // TC o(ns) // SC o(ns)
  int dp[1001][1001];
    int solve(int n, int a[], int s){
        if(n==-1){
            if(s==0) return 1;
            return 0;
        }
        if(s<0) return 0;
        if(s==0) return 1;
        if(dp[n][s]!=-1) return dp[n][s];
        return dp[n][s] = solve(n-1,a,s-a[n]) || solve(n-1,a,s);
    }
    int equalPartition(int N, int arr[]){
        int s =0 ;
        for(int i=0; i<N; i++) s+= arr[i];
        if(s&1) return 0;
        s=s/2;
        dp[N][s];
        memset(dp,-1,sizeof(dp));
        return solve(N-1,arr,s);
    }

  //7>> Friends Pairing Problem // TC o(n) // SC o(n)
  int mod  = 1000000007;
    long long int func(int n, long long int dp[]){
        if(n<=2) return n;
        if(dp[n]!=-1) return dp[n];

        long long int a = (((n-1)%mod)*func(n-2,dp)%mod)%mod;
        long long int b = func(n-1,dp)%mod;
        return dp[n] = a+b;
    }
    int countFriendsPairings(int n){
        long long int dp[n+1];
        memset(dp,-1,sizeof(dp));
        return func(n,dp)%mod;
    }

    // optimized SC(1) iterative approach
    int mod  = 1000000007;
    int count(int n){
      int a=1, b=2, c=3;
      if(n<=2) return n;
      for(long long int i=3; i<=n; i++){
        c = (b%mod + ((i-1)*a)%mod)%mod;
        a=b;
        b=c;
      }
      return c;
    }

//8>> Painting the Fence // TC o(n) // SC (n)
long long countWays(int n, int k){
        long long mod = 1000000007;
        if(n==0) return 0;
        if(n==1) return k;
        long long same = k%mod;
        long long diff = (k*(k-1))%mod;
        for(long long i=3; i<=n; i++){
            long long prev = diff%mod;
            diff = ((same+diff)*(k-1))%mod;
            same = prev%mod;
        }
        return (same+diff)%mod;
    }

//9>> cut length of a line segment each time is either x , y or z and maximize
// TC o(n) // SC o(n)
int dp[10005];
int solve(int n, int x, int y, int z){
    if(n==0) return 0;
    if(dp[n]!=-1) return dp[n];
    int op1=INT_MIN,op2=INT_MIN,op3=INT_MIN;
    if(n>=x) op1=solve(n-x,x,y,z);
    if(n>=y) op2=solve(n-y,x,y,z);
    if(n>=z) op3=solve(n-z,x,y,z);
    return dp[n] = 1+ max(op1,max(op2,op3));
}
int maximizeTheCuts(int n, int x, int y, int z){
    memset(dp,-1,sizeof(dp));
    int a = solve(n,x,y,z);
    if(a<0) return 0;
    else return a;
}

//10>> Longest Common Subsequence // TC O(|str1|*|str2|) // SC O(|str1|*|str2|)
int dp[1001][1001];
int solve(int x, int y, string s1, string s2){
  if(x==-1 or y==-1) return 0;
  if(dp[x][y]!=-1) return dp[x][y];
  if(s1[x]==s2[y]) return dp[x][y] = 1+solve(x-1,y-1,s1,s2);
  int a = solve(x-1,y,s1,s2);
  int b = solve(x,y-1,s1,s2);
  return dp[x][y] = max(a,b);
}
int lcs(int x, int y, string s1, string s2){
  memset(dp,-1,sizeof(dp));
  return solve(x-1,y-1,s1,s2);
}

//11>> Longest Repeated Subsequence // TC o(nn) // SC o(nn)
int n;
cin>>n;
string s;
cin>>s;
int dp[n+1][n+1];
for(int i=0; i<=n; i++){
    for(int j=0; j<=n; j++){
        if(i==0 || j==0)
        dp[i][j]=0;
        else if(s[i-1]==s[j-1] && i!=j)
        dp[i][j]=dp[i-1][j-1]+1;
        else dp[i][j]=max(dp[i-1][j],dp[i][j-1]);
    }
}
cout<<dp[n][n]<<endl;

//12>> Longest Increasing Subsequence // TC o(nlogn) // SC o(n)
int longestSubsequence(int n, int a[]){
   int dp[n+1];
   for(int i=1; i<=n; i++) dp[i]=INT_MAX;
   dp[0] = 0;
   for(int i=0; i<n; i++){
       int idx = upper_bound(dp,dp+n+1,a[i])-dp;
       if(a[i]>dp[idx-1] and a[i]<dp[idx]) dp[idx]=a[i];
   }
   int ma =0;
   for(int i=n; i>=0; i--){
       if(dp[i]!=INT_MAX) return i;
   }
   return ma;
}

//13>> Maximum sum increasing subsequence // TC o(nn) // SC o(n)
int maxSumIS(int a[], int n) {
	    int dp[n];
	    for(int i=0; i<n; i++){
	        dp[i] = a[i];
	    }
	    for(int i=0; i<n; i++){
	        for(int j=0; j<i; j++){
	            if(a[j]<a[i]) dp[i] = max(dp[i],dp[j]+a[i]);
	    }
	}
	int ma = 0;
	for(int i=0; i<n; i++) ma = max(ma,dp[i]);
	return ma;
	}

//14>> longest subsequence such that difference between adjacent elements is one
// TC o(nn) // SC o(n)
int longestSubsequence(int n, int a[]){
     int dp[n];
	    for(int i=0; i<n; i++) dp[i] = 1;
	    for(int i=0; i<n; i++){
	        for(int j=0; j<i; j++){
	            if(abs(a[j]-a[i])==1) dp[i] = max(dp[i],dp[j]+1);
	    }
	}
	int ma = 0;
	for(int i=0; i<n; i++) ma = max(ma,dp[i]);
	return ma;
	}

//15>> Maximum Length Chain of Pairs A pair (c, d) can follow
// another pair (a, b) if b < c// TC o(nn) // SC o(n)
bool cmp(struct val p, struct val q){
    return p.first < q.first;
}
int maxChainLen(struct val p[],int n){
     sort(p, p+n, cmp);
     vector<int>v(n,1);
     for(int i=1; i<n; i++){
         for(int j=0; j<i; j++){
             if(p[i].first > p[j].second && v[i] < v[j]+1){
                 v[i] = v[j]+1;
             }
         }
     }
     auto it = v.begin();
     int ans = INT_MIN;
     for(; it!= v.end(); it++) ans = max(ans, *it);
     return ans;
}
// Greedy approach // TC o(nlogn) // SC o(n)
bool cmp(const val&a,const val&b){
return(b.second>a.second);
}
int maxChainLen(struct val p[],int n){
sort(p,p+n,cmp);
int j=0,sum=1;
for(int i=0;i<n-1;i++){
    if(p[i+1].first>p[j].second){
    sum++;
    j=i+1;
   }
 }
return sum;
}

//16>> Maximum sum of pairs with specific difference // TC o(nlogn)
// Greedy approach
#include<bits/stdc++.h>
using namespace std;
int main(){
	int n,t,i,j,l,k,a[2000],sum;
	cin>>t;
	while(t--){
	    cin>>n;
	    sum=0;
	    for(i=0;i<n;i++)
	    cin>>a[i];
	    cin>>k;
	    sort(a,a+n,greater<int>());
	    for(i=0;i<n-1;i++){
	       if(a[i]-a[i+1]<k){
	           sum+=a[i]+a[i+1];
	           i++;
	       }
	    }
	    cout<<sum<<"\n";
	}
	return 0;
}

//17>> Egg Dropping Puzzle // TC o(nk) // SC o(nk)
int eggDrop(int e, int f) {
    int dp[f+1][e+1];
    for(int i = 1; i <= e; i++){
        dp[0][i] = 0;
        dp[1][i] = 1;
    }
    for(int i = 2; i <= f; i++)
        dp[i][1] = i;
    for(int i = 2; i <= f; i++){
        for(int j = 2; j <= e; j++){
            dp[i][j] = INT_MAX;
            for(int x = 1; x <= i; x++){
                dp[i][j] = min(dp[i][j], 1+max(dp[x-1][j-1], dp[i-x][j]));
            }
        }
    }
    return dp[f][e];
}

//18>> array cost[] represents the cost of ‘i’ kg packet of oranges,
// the task is to find the minimum cost to buy W kgs of oranges
// TC o(nn) // SC o(nn)
int dp[1001][1001];
	int solve(int a[], int n, int w){
	    if(n==-1 && w==0) return 0;
	    if(n==-1) return 1e9;
	    if(w==0) return 0;
	    if(dp[n][w]!=-1) return dp[n][w];
	    if((n+1)>w || a[n]==-1) return dp[n][w] = solve(a,n-1,w);
	    int x = solve(a,n-1,w);
	    int y = a[n] + solve(a,n,w-n-1);
	    return dp[n][w] = min(x,y);
	}
	int minimumCost(int cost[], int N, int W){
        memset(dp,-1,sizeof(dp));
        int x = solve(cost,N-1,W);
        if(x==1e9) return -1;
        return x;
	}

//19>> Longest Common Substring // TC o(mn) // SC o(mn)
int longestCommonSubstr (string s1, string s2, int n, int m){
        int dp[n+1][m+1];
        int res = 0;
        for(int i=0; i<=n; i++){
            for(int j=0; j<=m; j++){
                if(i==0 || j==0) dp[i][j] = 0;
                else if(s1[i-1]==s2[j-1]){
                    dp[i][j] = dp[i-1][j-1]+1;
                    res = max(res,dp[i][j]);
                }
                else dp[i][j] = 0;
            }
        }
        return res;
    }

//20>> Minimum removals from array to make max – min <= K
// TC o(nn) // SC o(nn)
int dp[MAX][MAX];
int solve(int a[], int i, int j, int k){
    if (i >= j) return 0;
    else if ((a[j] - a[i]) <= k) return 0;
    else if (dp[i][j] != -1) return dp[i][j];
    else if ((a[j] - a[i]) > k)
        dp[i][j] = 1 + min(solve(a,i+1,j,k),solve(a,i,j-1,k));
    return dp[i][j];
}
int removals(int a[], int n, int k){
    sort(a, a + n);
    memset(dp, -1, sizeof(dp));
    if (n == 1) return 0;
    else return solve(a, 0, n - 1, k);
}

// TC o(nlogn) // SC o(n)
int findInd(int key, int i,
            int n, int k, int arr[]){
    int start, end, mid, ind = -1;
    start = i + 1;
    end = n - 1;
    while (start < end){
        mid = start + (end - start) / 2;
        if (arr[mid] - key <= k){
            ind = mid;
            start = mid + 1;
        }
        else end = mid;
    }
    return ind;
}
int removals(int arr[], int n, int k){
    int i, j, ans = n - 1;
    sort(arr, arr + n);
    for (i = 0; i < n; i++){
        j = findInd(arr[i], i, n, k, arr);
        if (j != -1){
            ans = min(ans, n - (j - i + 1));
        }
    }
    return ans;
}
// upper bound function can be used in place of findInd;

//21>> Knapsack with Duplicate Items
// TC o(nw) // SC o(nw)
int dp[1001][1001];
int solve(int wt[], int val[], int w, int n){
    if(w==0 or n==-1) return 0;
    if(dp[n][w]!=-1) return dp[n][w];
    if(wt[n]>w) return dp[n][w] = solve(wt,val,w,n-1);
    int a,b;
    a = val[n] + solve(wt,val,w-wt[n],n);
    b = solve(wt,val,w,n-1);
    return dp[n][w] = max(a,b);
}
    int knapSack(int n, int w, int val[], int wt[]){
        dp[n][w];
        memset(dp,-1,sizeof(dp));
        return solve(wt,val,w,n-1);
    }

//22>> Longest Palindromic Subsequence
int lps(char *seq, int i, int j) {
if (i == j) return 1;
if (seq[i] == seq[j] && i + 1 == j) return 2;
if (seq[i] == seq[j])
    return lps (seq, i+1, j-1) + 2;
return max( lps(seq, i, j-1), lps(seq, i+1, j) );
}

// TC o(nn) // SC o(nn)
int lps(char *str) {
   int n = strlen(str);
   int i, j, cl;
   int L[n][n];
   for (i = 0; i < n; i++)
      L[i][i] = 1;
    for (cl=2; cl<=n; cl++) {
        for (i=0; i<n-cl+1; i++){
            j = i+cl-1;
            if (str[i] == str[j] && cl == 2)
               L[i][j] = 2;
            else if (str[i] == str[j])
               L[i][j] = L[i+1][j-1] + 2;
            else
               L[i][j] = max(L[i][j-1], L[i+1][j]);
        }
    }
    return L[0][n-1];
}

//23>> Longest alternating subsequence // TC o(n)
int AlternatingaMaxLength(vector<int>&a){
      int n = a.size();
      if(n==1) return 1;
      int dp[n][2];
      for(int i=0; i<n ;i++)
      dp[i][0] = dp[i][1] = 1;
      int ma  = 0;
      for(int i=1; i<n; i++){
          for(int j=0; j<i; j++){
              if(a[i]>a[j] && dp[i][0]<dp[j][1]+1)
              dp[i][0] = dp[j][1]+1;
              else if(a[i]<a[j] && dp[i][1]<dp[j][0]+1)
              dp[i][1] = dp[j][0]+1;
          }
          ma = max(ma,max(dp[i][0],dp[i][1]));
      }
      return ma;
  }

//24>> a player can pick x or y or 1 coin from n, where A always pics first
// whoever picks the last wins, for given if A wins or not // TC o(n)
bool findWinner(int x, int y, int n) {
    int dp[n + 1];
    dp[0] = false;
    dp[1] = true;
    for (int i = 2; i <= n; i++) {
        if (i - 1 >= 0 and !dp[i - 1])
            dp[i] = true;
        else if (i - x >= 0 and !dp[i - x])
            dp[i] = true;
        else if (i - y >= 0 and !dp[i - y])
            dp[i] = true;
        else dp[i] = false;
    }
    return dp[n];
}

//25>> Count Derangements (Permutation such that no element appears
// in its original position) // TC is expo
int countDer(int n){
  if (n == 1) return 0;
  if (n == 2) return 1;
  return (n - 1) * (countDer(n - 1) + countDer(n - 2));
}

// TC o(n) // SC o(n)
int countDer(int n){
    int der[n + 1] = {0};
    der[1] = 0;
    der[2] = 1;
    for (int i = 3; i <= n; ++i)
        der[i] = (i - 1) * (der[i - 1] + der[i - 2]);
    return der[n];
}

// TC o(n) // SC o(1)
int countDer(int n){
    if (n == 1 or n == 2) return n - 1;
    int a = 0;
    int b = 1;
    for (int i = 3; i <= n; ++i) {
        int cur = (i - 1) * (a + b);
        a = b;
        b = cur;
    }
    return b;
}

//26>> In each turn, a player selects either the first or last coin from
//the row, removes it from the row permanently, and receives the value
// of the coin determine the maximum possible amount of money you can
// win if you go first // TC o(nn) // SC o(nn)
int dp[1001][1001];
int solve(int i, int j, int a[]){
    if(i>j) return 0;
    if(dp[i][j]!=-1) return dp[i][j];
    int x = a[i] + min(solve(i+2,j,a),solve(i+1,j-1,a));
    int y = a[j] + min(solve(i,j-2,a),solve(i+1,j-1,a));
    return dp[i][j] = max(x,y);
}
long long maximumAmount(int arr[], int n) {
    dp[n][n];
    memset(dp,-1,sizeof(dp));
    return solve(0,n-1,arr);
}

//27>> Given a number N, the task is to find out the number of possible
// numbers of the given length // TC o(n) // SC o(n)
int mat[4][3] = {{1,2,3},
                 {4,5,6},
                 {7,8,9},
                 {-1,0,-1}};
    long long dp[101][101];
    long long solve(int i, int j, int n){
        if(n==1) return 1;
        if(dp[mat[i][j]][n]!=-1) return dp[mat[i][j]][n];
        long long a = solve(i,j,n-1);
        long long b,c,d,e;
        b=c=d=e=0;
        if(j-1>=0 and mat[i][j-1]!=-1) b = solve(i,j-1,n-1);
        if(j+1<3 and mat[i][j+1]!=-1) c = solve(i,j+1,n-1);
        if(i-1>=0 and mat[i-1][j]!=-1) d  = solve(i-1,j,n-1);
        if(i+1<4 and mat[i+1][j]!=-1) e = solve(i+1,j,n-1);
        return dp[mat[i][j]][n] = a+b+c+d+e;
    }
	long long getCount(int N){
		dp[10][N+1];
		memset(dp,-1,sizeof(dp));
		long long ans = 0;
		for(int i=0; i<4; i++){
		    for(int j=0; j<3; j++){
		        if(mat[i][j]!=-1)
		            ans += solve(i,j,N);
		    }
		}
		return ans;
	}

// for space optimized ans check gfg

//28>> Maximum sum Rectangle // TC o(rrc) // SC o(r)
int kadane(int arr[],int n){
    int sum=0;
    int maxi=INT_MIN;
    for(int i=0;i<n;i++){
        sum+=arr[i];
        maxi=max(maxi,sum);
        if (sum<0) sum=0;
    }
    return maxi;
}
void maxsum(int mat[c1][c2]){
    int ans=INT_MIN;
    int arr[row];

    for(int c1=0;c1<col;c1++){
        memset(arr,0,sizeof(arr));
        for(int c2=c1;c2<col;c2++){
            for(int i=0;i<row;i++){
                arr[i]+=mat[i][c2];
            }
            ans=max(ans,kadane(arr,row));
        }
    }
    cout<<ans<<endl;
}

//29>> Interleaved Strings, C is said to be interleaving A and B,
// if it contains all characters of A and B and order of all characters
// in individual strings is preserved // TC o(mn) // SC o(mn)
int dp[1001][1001];
bool solve(string A, string B, string C, int n, int m, int len){
    if(len==0) return 1;
    if(dp[n][m]!=-1) return dp[n][m];
    int a,b;
    a=b=0;
    if(n-1>=0 and A[n-1]==C[len-1]) a = solve(A,B,C,n-1,m,len-1);
    if(m-1>=0 and B[m-1]==C[len-1]) b = solve(A,B,C,n,m-1,len-1);
    return dp[n][m] = a or b;
}
bool isInterleave(string A, string B, string C) {
    int n = A.length();
    int m = B.length();
    int len = C.length();
    if(m+n != len) return 0;
    dp[n][m];
    memset(dp,-1,sizeof(dp));
    return solve(A,B,C,n,m,len);
}

//30>> Program for nth Catalan Number // TC o(nn) // SC o(n)
cpp_int findCatalan(int n){
    cpp_int dp[n+1];
dp[0]=1,dp[1]=1;
for(int i=2;i<=n;i++){
  dp[i]=0;
  for(int j=0;j<i;j++)
    dp[i]+=(dp[j]*dp[i-j-1]);
  }
return dp[n];
}

//31>> Matrix Chain Multiplication // TC o(nnn) // SC o(nn)
int dp[101][101];
    int solve(int arr[], int i, int j){
        if(i==j) return 0;
        if(dp[i][j]!=-1) return dp[i][j];
        dp[i][j] = INT_MAX;
        for(int k=i; k<j; k++){
            dp[i][j] = min(dp[i][j], solve(arr,i,k)+solve(arr,k+1,j)+
            arr[i-1]*arr[k]*arr[j]);
        }
        return dp[i][j];
    }
    int matrixMultiplication(int N, int arr[]){
        memset(dp, -1, sizeof dp);
        return solve(arr,1, N-1);
    }

  //32>> Gold Mine Problem // TC o(nm) // o(1)
  int maxGold(int n, int m, vector<vector<int>> M){
        for(int c=m-2; c>=0; c--){
            for(int r=0; r<n; r++){
                int right = M[r][c+1];
                int rup = (r==0)? 0:M[r-1][c+1];
                int rdown = (r==n-1)? 0:M[r+1][c+1];
                M[r][c] += max(right, max(rup,rdown));
            }
        }
        int res = M[0][0];
        for(int i = 1; i<n; i++)
            res = max(res,M[i][0]);
        return res;
    }

//33>> Assembly Line Scheduling // TC o(n) // SC o(n)
int carAssembly(int a[][NUM_STATION],int t[][NUM_STATION],
                int *e, int *x) {
    int T1[NUM_STATION], T2[NUM_STATION], i;
    T1[0] = e[0] + a[0][0];
    T2[0] = e[1] + a[1][0];
    for (i = 1; i < NUM_STATION; ++i) {
        T1[i] = min(T1[i - 1] + a[0][i],
                    T2[i - 1] + t[1][i] + a[0][i]);
        T2[i] = min(T2[i - 1] + a[1][i],
                    T1[i - 1] + t[0][i] + a[1][i]);
    }
    return min(T1[NUM_STATION - 1] + x[0],
               T2[NUM_STATION - 1] + x[1]);
}

// SC o(1)
int carAssembly(int a[][4],int t[][4],int *e, int *x) {
    int first, second, i;
    first = e[0] + a[0][0];
    second = e[1] + a[1][0];
    for(i = 1; i < 4; ++i) {
        int up =  min(first+a[0][i], second+t[1][i]+a[0][i]);
        int down = min(second+a[1][i], first+t[0][i]+a[1][i]);
        first = up;
        second = down;
    }
    return min(first + x[0], second + x[1]);
}

//34>> A Space Optimized Solution of LCS // TC o(n) // SC o(n)
int lcs(string &X, string &Y) {
    int m = X.length(), n = Y.length();
    int L[2][n + 1];
    bool bi;
    for (int i = 0; i <= m; i++) {
        bi = i & 1;
        for (int j = 0; j <= n; j++) {
            if (i == 0 || j == 0)
                L[bi][j] = 0;
            else if (X[i-1] == Y[j-1])
                 L[bi][j] = L[1 - bi][j - 1] + 1;
            else
                L[bi][j] = max(L[1 - bi][j], L[bi][j - 1]);
        }
    }
    return L[bi][n];
}

//bi can be also int bi = i%2;

//35>> LCS (Longest Common Subsequence) of three strings
// TC o(nnn) // SC o(nnn)
int n,m,z;
cin>>n>>m>>z;
string s,s1,s2;
cin>>s>>s1>>s2;
int dp[n+1][m+1][z+1];
memset(dp,0,sizeof dp);
for(int i=1;i<=n;i++){
  for(int j=1;j<=m;j++){
    for(int k=1;k<=z;k++){
      if(s[i-1]==s1[j-1]&&s[i-1]==s2[k-1])
          dp[i][j][k]=dp[i-1][j-1][k-1]+1;
      else
          dp[i][j][k]=max({dp[i][j-1][k],dp[i-1][j][k],dp[i][j][k-1]});
    }
  }
}
cout<<dp[n][m][z]<<endl;

// recursive dp
int dp[100][100][100];
memset(dp, -1,sizeof(dp));
int lcsOf3(int i, int j,int k) {
    if(i==-1||j==-1||k==-1) return 0;
    if(dp[i][j][k]!=-1) return dp[i][j][k];
    if(X[i]==Y[j] && Y[j]==Z[k])
        return dp[i][j][k] = 1+lcsOf3(i-1,j-1,k-1);
    else
        return dp[i][j][k] = max(max(lcsOf3(i-1,j,k),
                            lcsOf3(i,j-1,k)),lcsOf3(i,j,k-1));
}

// 36>> Count all subsequences having product less than K
// TC o(nk) // SC o(nk)
int productSubSeqCount(vector<int> &arr, int k) {
    int n = arr.size();
    int dp[k + 1][n + 1];
    memset(dp, 0, sizeof(dp));
    for (int i = 1; i <= k; i++) {
        for (int j = 1; j <= n; j++) {
            dp[i][j] = dp[i][j - 1];
            if (arr[j - 1] <= i && arr[j - 1] > 0)
                dp[i][j] += dp[i/arr[j-1]][j-1] + 1;
        }
    }
    return dp[k][n];
}

//37>> Maximum subsequence sum such that no three are consecutive
// TC o(n) // SC o(n)
int maxSumWO3Consec(int arr[], int n) {
    int sum[n];
    if (n >= 1) sum[0] = arr[0];
    if (n >= 2) sum[1] = arr[0] + arr[1];
    if (n > 2) sum[2] = max(sum[1], max(arr[1] + arr[2], arr[0] + arr[2]));
    for (int i = 3; i < n; i++)
        sum[i] = max(max(sum[i - 1], sum[i - 2] + arr[i]),
                     arr[i] + arr[i - 1] + sum[i - 3]);
    return sum[n - 1];
}

//recursive dp
int maxSumWO3(int n) {
    if(sum[n]!=-1) return sum[n];
    if(n==0) return sum[n] = 0;
    if(n==1) return sum[n] = arr[0];
    if(n==2) return sum[n] = arr[1]+arr[0];
    return sum[n] = max(max(maxSumWO3(n-1),maxSumWO3(n-2) + arr[n-1]),
                    arr[n-2] + arr[n-1] + maxSumWO3(n-3));
}

//38>> Smallest sum contiguous subarray // TC o(n)
int smallestSumSubarr(int arr[], int n) {
    int min_ending_here = INT_MAX;
    int min_so_far = INT_MAX;
    for (int i=0; i<n; i++) {
        if (min_ending_here > 0) min_ending_here = arr[i];
        else min_ending_here += arr[i];
        min_so_far = min(min_so_far, min_ending_here);
    }
    return min_so_far;
}

//40>> Largest Independent Set Problem // TC o(n) // SC o(n)
int LISS(node *root)  {
    if (root == NULL) return 0;
    if (root->liss) return root->liss;
    if (root->left == NULL && root->right == NULL)
        return (root->liss = 1);
    int liss_excl = LISS(root->left) + LISS(root->right);
    int liss_incl = 1;
    if (root->left)
        liss_incl += LISS(root->left->left) + LISS(root->left->right);
    if (root->right)
        liss_incl += LISS(root->right->left) + LISS(root->right->right);
    root->liss = max(liss_incl, liss_excl);
    return root->liss;
}

//41>> a player can score 3 or 5 or 10 points in a move. Given a total
//score n, find number of distinct combinations to reach the given score
// TC o(nnn) // SC o(n)
long long int count(long long int n){
	long long int table[n+1],i;
	memset(table, 0, sizeof(table));
	table[0]=1;
	for(int i=3;i<=n;i++)
       table[i]+=table[i-3];
       for(int i=5;i<=n;i++)
          table[i]+=table[i-5];
          for(int i=10;i<=n;i++)
              table[i]+=table[i-10];
	return table[n];
}

// optimized sol is like coin change Problem

//42>>  count the maximum number of balanced binary trees possible with
// height h. Print the result modulo 109 + 7 // TC o(h) // SC o(h)
long long int countBT(int h) {
        if(h == 0 or h == 1) return 1;
        long long int MOD = 1000000007;
        long long int dp[h + 1];
        dp[0] = dp[1] = 1;
        for(int i = 2; i <= h; i++)
            dp[i] = (dp[i - 1] * ((2 * dp[i - 2]) % MOD + dp[i - 1])) % MOD;
        return dp[h];
    }

//43>> Largest rectangular sub-matrix whose sum is 0 // TC o(nnn) // SC o(n)
int subsum(vector<int> a,int n){
        int s = 0;
        int ma = 1;
        unordered_map<int,int> m;
        for(int i=0;i<n;i++){
            s = s + a[i];
            if(s==0) ma = max(ma,i+1);
            else if(m[s]) ma = max(ma,i-m[s]+1);
            else m[s] = 1;
        }
        return ma;
}
void(int r, int c, vector<vector<int>> mat){
        int ma = INT_MIN;
        for(int i=0;i<r;i++){
            vector<int> ans(c);
            for(int j=i;j<r;j++){
                for(int col=0;col<c;col++){
                    ans[col] += mat[j][col];
                }
                ma = max(ma,subsum(ans,c)*(j-i+1));
            }
        }
        cout<<ma<<endl;
    }

//44>> Largest area rectangular sub-matrix with equal number of 1’s
// and 0’s [ IMP ]
same as above just replace 0 with 1;

//45>> Maximum profit by buying and selling a share at most twice [ IMP ]
// TC o(n) // SC o(n)
int mp(int p[], int n){
  int dp[n];
  int ma = dp[n-1];
  memset(dp,0,sizeof(dp));
  int mi = dp[0];
  for(int i = n-2; i>-1; i--){
    ma = max(p[i],ma);
    dp[i] = max(dp[i+1],ma-p[i]);
  }
  for(int j=1; j<n; j++){
    mi = min(mi,p[j]);
    dp[j] = (dp[j-1],dp[j]+p[j]-mi);
  }
  return dp[n-1];
}

//46>> longest Palindromic Substring // TC o(nn) // SC o(nn)
string longestPalindrome(string s) {
       if (s.empty()) return "";
       int n = s.size();
       std::vector<std::vector<bool> > dp(n, std::vector<bool>(n, false));
       int start = 0, len = 1;
       for (int i = 0; i < n; ++i) {
           dp[i][i] = true;
           for (int j = 0; j < i; ++j) {
               dp[j][i] = (s[i] == s[j]) && (i - j < 2 || dp[j+1][i-1]);
               if (dp[j][i] && i - j + 1 > len) {
                   len = i - j + 1;
                   start = j;
               }
           }
       }
       return s.substr(start, len);
   }

// TC o(nlogn) // SC o(1) expanding winddow
string longestPalindrome(string s) {
        if (s.empty()) return s;
        int idx = 0, len = 1;        int i = 0;
        while (i < s.size()) {
            int start = i;
            int end = i;
            while (end < s.size() - 1 && s[end] == s[end+1]) {
                end++;
            }
            i = end + 1;
            while (start > 0 && end < s.size() - 1 && s[start-1] == s[end+1]) {
                start--, end++;
            }
            if (end - start + 1 > len) {
                len = end - start + 1;
                idx = start;
            }
        }
        return s.substr(idx, len);
    }

//47>> Boolean Parenthesization Problem // TC o(nn) // SC o(nn)
int dp[2][202][202];
    int mod=1003;
    int Solve(string& X, int i, int j, bool isTrue){
    if (dp[isTrue][i][j] != -1)
    return dp[isTrue][i][j];
    if (i >= j){
    if (isTrue) dp[1][i][j] = X[i] == 'T';
    else dp[0][i][j] = X[i] == 'F';
    return dp[isTrue][i][j];
  }
    int ans = 0;
    for (int k = i + 1; k < j; k += 2) {
    int l_T = Solve(X, i, k - 1, true);
    int l_F = Solve(X, i, k - 1, false);
    int r_T = Solve(X, k + 1, j, true);
    int r_F = Solve(X, k + 1, j, false);
    if (X[k] == '|') {
    if (isTrue == true)
    ans += l_T * r_T + l_T * r_F + l_F * r_T;
    else ans += l_F * r_F;
  }
    else if (X[k] == '&'){
    if (isTrue == true) ans += l_T * r_T;
    else ans += l_T * r_F + l_F * r_T + l_F * r_F;
  }
    else if (X[k] == '^') {
    if (isTrue == true) ans += l_T * r_F + l_F * r_T;
    else ans += l_T * r_T + l_F * r_F;
  }
 }
   return dp[isTrue][i][j] = ans%mod;
}
    int countWays(int N, string S){
        memset(dp,-1,sizeof(dp));
        return Solve(S, 0, N-1, true);
    }

//48>> Palindrome PartitioningProblem // TC o(nn) // SC o(nn)
   int dp[501][501];
   bool isPalindrome(string str, int i, int j){
    while (i <= j){
        if (str[i++] != str[j--]) return false;
    }
    return true;
  }
    int fun(string str, int i, int j){
        if(dp[i][j]!=-1)return dp[i][j];
        if (i == j || isPalindrome(str, i, j))
            return dp[i][j]=0;
        int min = INT_MAX;
        for (int k = i; k <= j-1; k++){
            int right=0;
            if(isPalindrome(str,i,k)) right=fun(str,k+1,j);
            else{
                dp[i][k]=0;
                continue;
            }
            int count = 1 + right;
            if (count < min)
            min = count;
        }
        return dp[i][j]=min;
    }
    int palindromicPartition(string str){
        memset(dp,-1,sizeof(dp));
        int n=str.length();
        return fun(str,0,n-1);
    }

// NON Optimized
    int dp[501][501];
    bool ispalindrom(string s,int i,int j){
    while( i < j ){
    if( s[i++] != s[j--] )return false ;
    }
    return true;
    }
    int solve(string s,int i,int j){
    if(i>=j) return 0;
    if(dp[i][j]!=-1) return dp[i][j];
    if(ispalindrom(s,i,j)){
    dp[i][j] = 0;
    return 0;
    }
    int minimum=INT_MAX;
    for(int k=i;k<j;k++){
      int temp=solve(s,i,k)+solve(s,k+1,j)+1;
       if(minimum>temp) minimum = temp;
    }
    return dp[i][j]=minimum;
    }
    int palindromicPartition(string str){
    memset(dp,-1,sizeof(dp));
    return solve(str,0,str.size()-1);
    }

//49>> Word Wrap Problem // TC o(nn) // SC o(nn)
void print(int p[], int n){
    if(p[n]==1)
    cout<<p[n]<<" "<<n<<" ";
    else{
        print(p,p[n]-1);
        cout<<p[n]<<" "<<n<<" ";
    }
}
void word(int a[], int n, int k){
    int space[n+1][n+1];
    int ls[n+1][n+1];
    int c[n+1];
    int p[n+1];
    for(int i=1; i<=n; i++){
        space[i][i]=k-a[i];
        for(int j=i+1; j<=n; j++){
            space[i][j]=space[i][j-1]-a[j]-1;
        }
    }
    for(int i=1; i<=n; i++){
        for(int j=i; j<=n; j++){
            if(space[i][j]<0){
                ls[i][j]=INT_MAX;
            }
            else if(j==n && space[i][j]>=0){
                ls[i][j]=0;
            }
            else{
                ls[i][j]=space[i][j]*space[i][j];
            }
        }
    }
    c[0]=0;
    for(int i=1; i<=n; i++){
        c[i]=INT_MAX;
        for(int j=1; j<=i; j++){
            if(c[j-1] != INT_MAX && ls[j][i] != INT_MAX && c[j-1]+ls[j][i]<c[i]){
                c[i]=c[j-1]+ls[j][i];
                p[i]=j;
            }
        }
    }
    print(p,n);
    cout<<endl;
}

//50>> Maximum Length of Pair Chain // Greedy // TC o(nlogn)
      struct val{
    	int first;
    	int second;
    };
    bool cmp(const val&a,const val&b){
    return(b.second>a.second);
    }
    int findLongestChain(vector<vector<int>>& pairs) {
        int n = pairs.size();
        val arr[n];
        for(int i=0; i<n; i++){
            arr[i].first = pairs[i][0];
            arr[i].second = pairs[i][1];
        }
        sort(arr,arr+n,cmp);
        int j=0,sum=1;
        for(int i=0;i<n-1;i++){
            if(arr[i+1].first>arr[j].second){
            sum++;
            j=i+1;
           }
         }
        return sum;
    }

//51>> Maximum profit by buying and selling a share at most k times
// TC o(nk) // SC o(nk)
int maxProfit(int K, int N, int A[]){
        int dp[K+1][N];
        for(int i = 0; i <= K; i++)
            dp[i][0] = 0;
        for(int j = 0; j < N; j++)
            dp[0][j] = 0;
        for(int i = 1; i <= K; i++){
            int maxi = INT_MIN;
            for(int j = 1; j < N; j++){
                maxi = max(maxi, dp[i-1][j-1] - A[j-1]);
                dp[i][j] = max(maxi + A[j], dp[i][j-1]);
            }
        }
        return dp[K][N-1];
    }

//52>> Maximum path sum in matrix // TC o(nn) // SC o(nn)
int maximumPath(int N, vector<vector<int>> m){
    int n=N;
    int dp[n][n];
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            if(i==0){
                dp[i][j]= m[i][j];}
            else if(j>=1 && j+1<n)
               dp[i][j]=max(dp[i-1][j-1],max(dp[i-1][j+1],dp[i-1][j])) + m[i][j];
            else if(j==0 )
               dp[i][j]=max(dp[i-1][j+1],dp[i-1][j]) + m[i][j];
            else if(j==n-1 )
               dp[i][j]=max(dp[i-1][j-1],dp[i-1][j]) + m[i][j];
        }
    }
    int k=0;
    for(int i=0;i<n;i++){
        if(k<dp[n-1][i])
        k=dp[n-1][i];
    }
    return k;
   }

//53>> Largest square formed in a matrix // TC o(nn) // SC o(nn)
// Utilizing same input matrix
int maxSquare(int M, int N, vector<vector<int>> t){
        int mx = 0 ;
        for( int i = 1 ; i < M ; i++ )
        for( int j = 1 ; j < N ; j++ )
        if(t[i][j]) t[i][j] = 1 + min({t[i-1][j-1], t[i-1][j], t[i][j-1]});
        for( int i = 0 ; i < M ; i++ )
        for( int j = 0 ; j < N ; j++ )
        mx = max( mx , t[i][j] );
        return mx ;
    }

//54>> Optimal bst // TC o(nnn) // SC o(nn)
int optimalSearchTree(int keys[], int freq[], int n)  {
    int cost[n][n];
    for (int i = 0; i < n; i++)
        cost[i][i] = freq[i];
    for (int L = 2; L <= n; L++){
        for (int i = 0; i <= n-L+1; i++)  {
            int j = i+L-1;
            cost[i][j] = INT_MAX;
            for (int r = i; r <= j; r++){
            int c = ((r > i)? cost[i][r-1]:0) +
                    ((r < j)? cost[r+1][j]:0) +
                    sum(freq, i, j);
            if (c < cost[i][j]) cost[i][j] = c;
            }
        }
    }
    return cost[0][n-1];
}

// for storin bst We can create another auxiliary array of size n to store
// the structure of tree. All we need to do is, store the chosen ‘r’ in the
// innermost loop
