//ARRAYS

//1>> Reverse the array  // TC o(n) //SC o(1)

string reverseWord(string str){
  string s="";
  for(int i=str.length()-1;i>=0;i--){
    s+=str[i];}
  return s;
}

//2>> Middle of Three  // TC o(1) // SC o(1)

int middle(int A, int B, int C){
int x = A-B;
int y = B-C;
int z = C-A;
if(x*y>0) return B;
if(y*z>0) return C;
return A;
  }

//3>> Kth smallest element // TC o(nlogn) // SC o(1)
int kthSmallest(int arr[], int l, int r, int k) {
    sort(arr,arr+(r+1));
    return arr[k-1];
}

// TC o(n+klogn) SC(n)
priority queue meathod

// TC o(n) // SC o(n)
int kthSmallest(int arr[], int l, int r, int k) {
   bool flag[100005];
   memset(flag,false,sizeof(flag));
      for(int i=l;i<=r;i++) flag[arr[i]]=true;
      for(int i=1;i<=100005;i++){
          if(flag[i]==true){
              k--;
          if(!k)
              return i;
          }
      }
     return -1;
}

//4>> Sort an array of 0s, 1s and 2s // TC o(n) // SC o(1)
void sort012(int a[], int n){
    int c=0;
    for(int i=0; i<n; i++){
        if(a[i]==0){
            swap(a[c],a[i]);
            c++;
        }
    }
    for(int i=c; i<n; i++){
        if(a[i]==1){
            swap(a[c],a[i]);
            c++;
        }
    }
}

//5>> Move all negative numbers to beginning and positive to end // TC o(n)
void rearrange(int arr[], int n){
    int j = 0;
    for (int i = 0; i < n; i++) {
        if (arr[i] < 0) {
            if (i != j) swap(arr[i], arr[j]);
            j++;
        }
    }
}

//6>> Size of Union of two arrays // TC o((n+m)log(n+m)) // SC o(m+n)
unordered_set<int> s;
cin>>n>>m;
int a[n],b[m];
for(int j=0; j<n; j++){
    cin>>a[j];
    s.insert(a[j]);
}
for(int k=0; k<m; k++){
    cin>>b[k];
    s.insert(b[k]);
}
cout<<s.size()<<endl;

//7>> Cyclically rotate an array by one // TC o(n)
int s = a[n-1];
for(int i=n-1; i>0; i--){
    a[i]=a[i-1];
}
a[0]=s;

//8>> Kadane's Algorithm // TC o(n)
int maxSubarraySum(int arr[], int n){
    int tillnow = arr[0];
    int curr = arr[0];
for (int i = 1; i < n; i++){
curr = max(arr[i], curr+arr[i]);
tillnow = max(tillnow, curr);
}
return tillnow;
}

//9>> Minimise the maximum difference between heights // TC O(NlogN)
int getMinDiff(int arr[], int n, int k) {
sort(arr, arr+n);
int minEle, maxEle;
int result = arr[n-1] - arr[0];
for(int i =1; i<=n-1; i++){
if(arr[i]>=k){
maxEle = max(arr[i-1] + k, arr[n-1]-k);
minEle = min(arr[0]+k, arr[i]-k);
result = min(result, maxEle-minEle);
}
else continue;
}
return result;
}

//10>> Minimum no. of Jumps to reach end of an array // TC o(n)
int minJumps(int arr[], int n){
    if(n<=0) return 0;
    int steps=arr[0];
    int maxi=arr[0];
    int jumps=1;
    for(int i=1;i<n;i++){
        if(i==n-1) return jumps;
        maxi=max(maxi,arr[i]+i);
        steps--;
        if(steps==0){
            jumps++;
            steps=maxi-i;
        }
        if(steps<=0) return -1;
    }
    return jumps;
}

//11>> find duplicate in an array of N+1 Integers //TC o(n)
int findDuplicate(vector<int>& nums) {
   int s=nums.size();
        vector<int> ans(s,0);
        for(int i=0;i<s;i++){
            ans[nums[i]]++; //counting the frequency of elements
            if(ans[i]>=2) return i; //if frequency >= 2 then return
        }
          for(int i=0;i<s;i++){
              if(ans[i]>=2) return i; //if frequency >= 2 then return
        }
        return 0;
    }

//11>> Merge 2 sorted arrays without using Extra space //TC o(nlogn)
void merge(int arr1[], int arr2[], int n, int m) {
int i=n-1,j=0;
while(i>=0 && j<m) {
if(arr1[i]>=arr2[j]){
  int temp=arr1[i];
  arr1[i]=arr2[j];
  arr2[j]=temp;
 }
i--;j++;
}
sort(arr1,arr1+n);
sort(arr2,arr2+m);
}

//12>> Next Permutation // TC o(nn)
void findnext(vector<int>& nums,int i){
        reverse(nums.begin()+i+1,nums.end());
        for(int k=i+1;k<nums.size();k++)
            if(nums[k]>nums[i]){
                swap(nums[i],nums[k]);
                return;
            }
    }
    void nextPermutation(vector<int>& nums) {
        for(int i=nums.size()-1;i>=1;i--){
            if(nums[i]>nums[i-1]){
                findnext(nums,i-1);
                return;
            }
        }
        reverse(nums.begin(),nums.end()); //end condition
    }

    //13>> Count inversion // TC o(n)
    long long merge(long long arr[], long long temp[], long long left,
          long long mid, long long right) {
    long long i, j, k;
    long long inv_count = 0;
    i = left;
    j = mid;
    k = left;
    while ((i <= mid - 1) && (j <= right)) {
        if (arr[i] <= arr[j]) temp[k++] = arr[i++];
        else {
            temp[k++] = arr[j++];
            inv_count = inv_count + (mid - i);
        }
    }
    while (i <= mid - 1) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];
    for (i = left; i <= right; i++) arr[i] = temp[i];
    return inv_count;
}
long long _mergeSort(long long arr[],long long temp[],long long left,long long right){
    long long mid, inv_count = 0;
    if (right > left) {
        mid = (right + left) / 2;
        inv_count += _mergeSort(arr, temp, left, mid);
        inv_count += _mergeSort(arr, temp, mid + 1, right);
        inv_count += merge(arr, temp, left, mid + 1, right);
    }
    return inv_count;
}

long long int inversionCount(long long arr[], long long N){
   long long temp[N];
   return _mergeSort(arr, temp, 0, N - 1);
}

//14>> Best time to buy and Sell stock // TC o(n) // SC o(1)
int maxProfit(vector<int>& prices){
     if (prices.size() <= 1) return 0;
int size = prices.size();
int iterator, maxProfit = 0;
int currentPointer = 1;
int lowestPricePointer = 0; //var to keep track of the lowest element
for(iterator = 1; iterator < size; iterator++){
if (prices[currentPointer] > prices[lowestPricePointer]) {
maxProfit = max(prices[currentPointer] - prices[lowestPricePointer], maxProfit);
       }
else lowestPricePointer = currentPointer;
currentPointer += 1;
   }
  return maxProfit;
}

//15>> Count pairs with given sum // TC o(n) // SC o(n)
int getPairsCount(int arr[], int n, int k) {
    int c=0;
    unordered_map<int,int> m;
    for(int i=0;i<n;i++){
        if(m.find(k-arr[i])!=m.end()) c+=m[k-arr[i]]; //if duplicates exist then it is counted twice as the pair would repeat
        m[arr[i]]++;//anyways storing the element
    }
    return c;
  }

//16>> find common elements In 3 sorted arrays // TC o(n1+n1+n3)
// SC o(n1+n2+n3)
vector<int> commonElements(int A[], int B[], int C[], int n1, int n2, int n3){
            map<int,int> hashmap;
            vector<int> vec;
            hashmap[A[0]]++;
            for(int i=1;i<n1;i++)
                if(A[i-1] != A[i]) hashmap[A[i]]++;
            hashmap[B[0]]++;
            for(int i=1;i<n2;i++)
                if(B[i-1] != B[i]) hashmap[B[i]]++;
            hashmap[C[0]]++;
            if(hashmap[C[0]] == 3) vec.push_back(C[0]);
            for(int i=1;i<n3;i++){
                if(C[i-1] != C[i]){
                  hashmap[C[i]]++;
                    if(hashmap[C[i]] == 3) vec.push_back(C[i]);
                }
            }
            return vec;
        }

//17>> Rearrange the array in alternating positive and negative items with O(1) extra space
// TC o(n)
void rearrange(int arr[], int n){
    int i = -1, j = n;
    while (i < j){
        while(i <= n - 1 and arr[i] > 0) i += 1;
        while (j >= 0 and arr[j] < 0) j -= 1;
        if (i < j) swap(arr[i], arr[j]);
    }
    if (i == 0 || i == n) return;
    int k = 0;
    while (k < n && i < n){
        swap(arr[k], arr[i]);
        i = i + 1;
        k = k + 2;
    }
}

//18>> Find if there is any subarray with sum equal to 0 // TC o(n) // SC o(n)
bool subArrayExists(int arr[], int n){
    int sum = 0;
    map<int,bool> u;
    for(int i =0; i<n; i++){
        sum+=arr[i];
        if(sum==0 || u.count(sum)==true) return true;
        else u[sum] = true;
    }
    return false;
  }

//19>> Find factorial of a large number // TC o(nn) // SC o(n)
cin>>n;
    vector<int> v;
    v.push_back(1);
    for(int i=2; i<=n; i++){
        int carry =0;
        for(int j=0; j<v.size(); j++){
            int mul = v[j]*i+carry;
            v[j]=mul%10;
            carry = mul/10;
        }
        while(carry){
            v.push_back(carry%10);
            carry = carry/10;
        }
    }
    for(int i=v.size()-1; i>-1; i--){
        cout<<v[i];
    }

//20>> find maximum product subarray // TC o(n) // SC o(1)
long long maxProduct(int *arr, int n) {
      long long ans,maxp,minp;
      ans=maxp=minp=arr[0];
      for(int i=1;i<n;i++){
          if(arr[i]<0)
              swap(maxp,minp);
          maxp =max<long long int>(maxp*arr[i],arr[i]);
          minp =min<long long int>(minp*arr[i],arr[i]);
          ans = max(ans,maxp);
      }
      return ans;
}

//21>> Longest consecutive subsequence // TC o(n) // SC o(n)
int findLongestConseqSubseq(int arr[], int N){
     vector<int> hash(100001,0);
  for(int i=0;i<N;i++) hash[arr[i]]=1;
  int i=0;int max=1;
  while(i<100001){
    int cnt=0;
    while(i<100001 && hash[i]!=0){
        i++;cnt++;
    }
    if(cnt>max) max=cnt;
    while(i<100001 && hash[i]==0) i++;
  }
  return max;
 }

//22>> Find whether an array is a subset of another array // TC o(n) // SC o(n)
bool arraySubset(int arr1[], int arr2[], int m, int n){
    unordered_set<int> s;
    for(int i = 0; i < m; i++) s.insert(arr1[i]);
    for(int i = 0; i < n; i++){
        if(s.find(arr2[i]) == s.end()) return false;
    }
    return true;
}

//23>> Find the triplet that sum to a given value // TC o(nlogn) // SC o(1)
bool doubleSum(int arr[], int l, int h, int x){
    while(l < h){
        if(arr[l] + arr[h] == x) return true;
        if(arr[l] + arr[h] > x) h--;
        else l++;
    return false;
   }
 }

bool tripletSumArray(int arr[], int n, int x){
    sort(arr, arr+n);
    for(int i = n-1; i >= 2; i--)
        if(doubleSum(arr, 0, i-1, x - arr[i]) == true) return true;
    return false;
}

//24>> Trapping Rain water problem // TC o(n)
int n;
cin>>n;
int arr[n];
for(int i=0; i<n; i++) cin>>arr[i];
int l=0; int h=n-1;
int lh=0; int rh=0; int ans=0;
while(l<h){
    if(arr[l]<arr[h]){
        if(arr[l]>lh) lh=arr[l];
        else ans+=lh-arr[l];
        l++;
    }
    else{
        if(arr[h]>rh) rh=arr[h];
        else ans+=rh-arr[h];
        h--;
    }
}
cout<<ans<<endl;

//25>> The difference between the number of chocolates given to the students
// having packet with maximum chocolates and student having packet with minimum
// chocolates is minimum. // TC o(n)
int n,m;
	    cin>>n;
	    int arr[n];
	    for(int i=0; i<n; i++) cin>>arr[i];
	    cin>>m;
	    int mn = INT_MAX;
	    sort(arr,arr+n);
	    for(int i=0; i+m-1<n; i++)
	        if(arr[i+m-1]-arr[i]<mn) mn=arr[i+m-1]-arr[i];
	    cout<<mn<<endl;

//26>> Smallest subarray with sum greater than x // TC o(nn)
int n,x;
cin>>n>>x;
int arr[n];
for(int i=0; i<n; i++) cin>>arr[i];
int end=0; int st=0;
int sum=0; int ans=INT_MAX;
while(end!=n){
    sum+=arr[end++];
    while(sum>x){
        ans = min(ans,end-st);
            sum-=arr[st++];
    }
}
cout<<ans<<endl;

//27>> Three way partitioning of an array around a given value // TC o(n)
void threeWayPartition(vector<int>& A,int a, int b){
   int low = 0, mid = 0, high = A.size()-1;
  while(mid <= high){
    if(A[mid] < a){
      swap(A[low], A[mid]);
      low++; mid++;
    }
    else if(A[mid] > b){
      swap(A[high], A[mid]);
      high--;
    }
    else mid++;
  }
}

//28>> Minimum swaps and K together // TC o(n)
int n,k;
cin>>n;
int arr[n];
for(int i=0; i<n; i++) cin>>arr[i];
cin>>k;
int good=0;
int bad=0;
int ans=INT_MAX;
for(int i=0; i<n; i++)
    if(arr[i]<=k) good++;
for(int i=0; i<good; i++)
    if(arr[i]>k) bad++;
int i=0; int j=good-1;
while(j<n){
    ans = min(ans,bad);
    j++;
    if(j<n && arr[j]>k) bad++;
    if(arr[i]>k) bad--;
    i++;
}
if(ans==INT_MAX) cout<<"0"<<endl;
else cout<<ans<<endl;

//29>> Median of 2 sorted arrays of equal size
// TC o(n)
int getMedian(int ar1[], int ar2[], int n) {
    int i = 0; /* Current index of  i/p array ar1[] */
    int j = 0; /* Current index of  i/p array ar2[] */
    int count;
    int m1 = -1, m2 = -1;
    for (count = 0; count <= n; count++){
        if (i == n) {
            m1 = m2;
            m2 = ar2[0];
            break;
        }
        else if (j == n) {
            m1 = m2;
            m2 = ar1[0];
            break;
        }
        if (ar1[i] <= ar2[j]){
            m1 = m2;
            m2 = ar1[i];
            i++;
        }
        else{
            m1 = m2;
            m2 = ar2[j];
            j++;
        }
    }
    return (m1 + m2)/2;
}

// TC o(logn)
int median(int arr[], int n) {
    if (n % 2 == 0)
        return (arr[n / 2] + arr[n / 2 - 1]) / 2;
    else
        return arr[n / 2];
}

int getMedian(int ar1[], int ar2[], int n){
    if (n <= 0) return -1;
    if (n == 1) return (ar1[0] + ar2[0]) / 2;
    if (n == 2) return (max(ar1[0], ar2[0]) + min(ar1[1], ar2[1])) / 2;
    int m1 = median(ar1, n);
    int m2 = median(ar2, n);
    if (m1 == m2) return m1;
    if (m1 < m2) {
        if (n % 2 == 0)
            return getMedian(ar1 + n / 2 - 1, ar2, n - n / 2 + 1);
            return getMedian(ar1 + n / 2, ar2, n - n / 2);
    }
    if (n % 2 == 0)
        return getMedian(ar2 + n / 2 - 1, ar1, n - n / 2 + 1);
        return getMedian(ar2 + n / 2, ar1, n - n / 2);
}

//30>> Median of 2 sorted arrays of different size // TC O(log(min(n,m)))
int findMedian(int arr[], int n, int brr[], int m){
    int begin1 = 0 ; int end1 = n;
    while(begin1<=end1){
        int i1 = (begin1+end1)/2;
        int i2 = (n+m+1)/2 - i1;
        int min1 = (i1==n) ? INT_MAX : arr[i1];
        int max1 = (i1==0) ? INT_MIN : arr[i1-1];
        int min2 = (i2==m) ? INT_MAX : brr[i2];
        int max2 = (i2==0) ? INT_MIN : brr[i2-1];
        if((max1<=min2) && (max2<=min1) ){
            if((n+m)%2==0) return ((double) (max(max1,max2) + min(min1, min2)) / 2) ;
            else return ((double) max(max1, max2));
        else if(max1>min2) end1 = i1-1;
        else begin1 = i1+1;
        }
    }
}

//31>> find all elements that appear more than n/k times // TC o(n) // SC o(n)
void morethanNbyK(int arr[], int n, int k){
    int x = n / k;
    unordered_map<int, int> freq;
    for(int i = 0; i < n; i++)
        freq[arr[i]]++;
    for(auto i : freq)
        if (i.second > x) cout << i.first << endl;
        }
    }
}

//32>> Maximum profit by buying and selling a share at most twice
// TC o(n) // SC o(n)
int maxProfit(vector<int>& a){
        int n = a.size();
        int dp[n];
        for(int i=0; i<n; i++) dp[i] = 0;
        int ma = a[n-1];
        int mi = a[0];
        for(int i= n-2; i>=0; i--){
            if(a[i]>ma) ma = a[i];
            dp[i] = max(dp[i+1],ma-a[i]);
        }
        for(int i=1; i<n; i++){
            if(a[i]<mi) mi=a[i];
            dp[i] = max(dp[i-1],dp[i]+(a[i]-mi));
        }
        return dp[n-1];
    }
