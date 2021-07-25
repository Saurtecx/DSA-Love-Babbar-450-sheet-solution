
// Searching and Sorting

//1>> find indexes of first and last occurrences of an element x
// in the given array // TC o(n)
int n,k,f=-1;
int b=1;
cin>>n>>k;
int arr[n];
for(int i=0; i<n; i++){
    cin>>arr[i];
    if(arr[i]==k){
        if(b){
        cout<<i<<" ";
        b--;
        }
        f=i;
    }
}
cout<<f<<endl;

//2>> find the elements whose value is equal to that of its index value
// TC o(n)
vector<int> valueEqualToIndex(int arr[], int n) {
   vector<int> x;
   for(int i=0; i<n; i++){
       if(arr[i]==i+1) x.push_back(i+1);
   }
   return x;
}

//3>> You are given an integer array nums sorted and  rotated at some
// pivot unknown to you beforehand // TC o(logn)
int search(vector<int>& nums, int target) {
        int n = nums.size();
        int l = 0;
        int r = n-1;
        int mid;
        while(l<=r){
            mid=(l+r)/2;
            if(nums[mid]==target) return mid;
            if(nums[mid]>=nums[l]){
                if(target>=nums[l] && target<=nums[mid]) r=mid-1;
                else l=mid+1;
            }
            else{
                if(target>=nums[mid] && target<=nums[r])
                    l=mid+1;
                else r=mid-1;
            }
        }
        return -1;
    }

//4>> You are given a number N, you have to output the number of integers
// less than N in the sample space S, all perfect squares starting from
// 1, 4, 9 and so on // TC o(sqrt(n))
int countSquares(int N){
      return pow(N-1,.5);
      }

//5>> Given three distinct numbers A, B and C. Find the number with
// value in middle // TC o(1)
int middle(int A, int B, int C){
        int x = A-B;
        int y = B-C;
        int z = C-A;
        if(x*y>0) return B;
        if(y*z>0) return C;
        return A;
    }

//6>> find a point on given line for which sum of distances from given
// set of points is minimum // TC (nlognbase3)
struct point {
    int x, y;
    point()  {}
    point(int x, int y) : x(x), y(y)   {}
};

//  structure defining a line of ax + by + c = 0 form
struct line {
    int a, b, c;
    line(int a, int b, int c) : a(a), b(b), c(c) {}
};
double dist(double x, double y, point p) {
    return sqrt(sq(x - p.x) + sq(y - p.y));
}
/*  Utility method to compute total distance all points
    when choose point on given line has x-cordinate value as X   */
double compute(point p[], int n, line l, double X) {
    double res = 0;
    double Y = -1 * (l.c + l.a*X) / l.b;
    for (int i = 0; i < n; i++)
        res += dist(X, Y, p[i]);
    return res;
}
//  Utility method to find minimum total distance
double findOptimumCostUtil(point p[], int n, line l) {
    double low = -1e6;
    double high = 1e6;
    while ((high - low) > EPS) {
        double mid1 = low + (high - low) / 3;
        double mid2 = high - (high - low) / 3;
        double dist1 = compute(p, n, l, mid1);
        double dist2 = compute(p, n, l, mid2);
        if (dist1 < dist2)
            high = mid2;
        else
            low = mid1;
    }
    return compute(p, n, l, (low + high) / 2);
}
//  method to find optimum cost
double findOptimumCost(int points[N][2], line l) {
    point p[N];
    for (int i = 0; i < N; i++)
        p[i] = point(points[i][0], points[i][1]);
    return findOptimumCostUtil(p, N, l);
}

//7>> Given an unsorted array Arr of size N of positive integers. One
// number 'A' from set {1, 2, …N} is missing and one number 'B' occurs
// twice in array. Find these two numbers // TC o(n) // SC o(n)
int *findTwoElement(int *arr, int n) {
        map<int,int>mp;
        for(int i=0;i<n;i++)
            mp[arr[i]]++;
        int *ans=new int(2);
        for(auto it=mp.begin();it!=mp.end();it++){
            if(it->second>1){
                ans[0]=it->first;
                break;
            }
        }
        for(int i=1;i<=n;i++){
            if(mp.find(i)==mp.end())
                ans[1]=i; break;
        }
        return ans;
    }
// 1. Count sort
// 2. Mark visited element as negative
// 3. HashMap
// 4. two equations
// first eq: (sum of 1 to N) - (sum of arr) = x-y
// sec eq: (product of 1 to N) / (product of arr) = x/y
// 5. XOR: xor of (1 to N) and arr = x^y

//8>> find element that appears more than N/2 times in the array
// TC o(n) // SC o(1)
int majorityElement(int arr[], int size){
    if(size == 1) return arr[0];
    if(size == 2) return -1;
    int count = 1, res = arr[0];
    for(int i = 1; i < size; i++){
        if(arr[i] == res) count++;
        else count--;
        if(count == 0){
            res = arr[i];
            count = 1;
        }
    }
    count = 0;
    for(int i = 0; i < size; i++)
    if(arr[i] == res) count++;
    if(count > size/2) return res;
    return -1;
}

//9>> Searching in an array where adjacent differ by at most k
int search(int arr[], int n, int x, int k){
    int i = 0;
    while (i < n){
        if (arr[i] == x) return i;
        i = i + max(1, abs(arr[i]-x)/k);
    }
    cout << "number is not present!";
    return -1;
}

//10>> Given an unsorted array Arr[] and a number N. You need to write
// a program to find if there exists a pair of elements in the array
// whose difference is N
int n,x;
	    cin>>n>>x;
	    int arr[n];
	    for(int i=0; i<n; i++) cin>>arr[i];
	    unordered_map<int,int> m;
	    int f=-1;
	    for(int i=0; i<n; i++)  m[arr[i]] =1;
	    for(int i=0; i<n; i++){
	        if(m[arr[i]+x]){
	            f=1;
	            break;
	        }
	    }
	    cout<<f<<endl;

//11>> Find all the unique quadruple from the given array that sums up to
// the given number // TC
vector<vector<int> > fourSum(vector<int> &arr, int k) {
    vector<vector<int>> res;
    if(arr.empty()) return res;
    int n =arr.size();
    sort(arr.begin(),arr.end());
    for(int i=0; i<n; i++){
        for(int j=i+1; j<n; j++){
            int t = k -arr[i]-arr[j];
            int f = j+1;
            int b = n-1;
            while(f<b){
                int ts = arr[f]+arr[b];
                if(ts<t) f++;
                else if(ts>t) b--;
                else{
                    vector<int> qd(4,0);
                    qd[0] = arr[i];
                    qd[1] = arr[j];
                    qd[2] = arr[f];
                    qd[3] = arr[b];
                    res.push_back(qd);
                    while(f<b && qd[2]==arr[f]) f++;
                    while(f<b && qd[3]==arr[b]) b--;
                    while(j+1<n && arr[j+1]==arr[j]) j++;
                    while(i+1<n && arr[i+1]==arr[i]) i++;
                }
            }
        }
    }
    return res;
}

// TC o(nn) // SC o(nn)
void findFourElements(int arr[], int n, int X){
    unordered_map<int, pair<int, int> > mp;
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            mp[arr[i] + arr[j]] = { i, j };
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            int sum = arr[i] + arr[j];
            if (mp.find(X - sum) != mp.end()) {
                pair<int, int> p = mp[X - sum];
                if (p.first != i && p.first != j
                    && p.second != i && p.second != j) {
                    cout << arr[i] << ", " << arr[j] << ", "
                         << arr[p.first] << ", "<< arr[p.second];
                    return;
                }
            }
        }
    }
}

//12>> n houses in a single line a looter when looting the houses he
// will never loot two consecutive houses, maximize // TC o(n) // SC o(n)
ll dp[10001];
ll solve(ll i, ll a[]){
    if(i<=-1) return 0;
    if(dp[i]!=-1) return dp[i];
    ll op1 = a[i]+solve(i-2,a);
    ll op2 = solve(i-1,a);
    return dp[i] = max(op1,op2);
}
ll FindMaxSum(ll arr[], ll n){  dp[n];
   memset(dp,-1,sizeof(dp));
   return solve(n-1,arr);
}

//13>> an array arr[] of distinct integers of size N and a sum value X,
// the task is to find count of triplets with the sum smaller than the
// given sum value X // TC  o(nn)
long long countTriplets(long long arr[], int n, long long sum){
	    long long ans = 0;
	    sort(arr,arr+n);
      for (int i = 0; i < n-2; i++){
       int j = i + 1, k = n - 1;
       while (j < k){
       if (arr[i] + arr[j] + arr[k] >= sum) k--;
       else{
        ans += (k - j);
        j++;
       }
     }
    }
     return ans;
	}

//14>> Merge Without Extra Space o(m+n +nlogn +mlogm)
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

//15>> Zero Sum Subarrays // TC o(n) // SC o(n)
ll findSubarray(vector<ll> arr, int n ) {
    ll sum = 0;
	unordered_map<int, int>mp;
	mp[0] = 1;
	int temp = 0;
	for (int i = 0; i < n; i++) {
		temp += arr[i];
		if (mp.find(temp) != mp.end()) {
			sum += mp[temp];
			mp[temp]++;
		}
		else mp[temp]++;
	}
	return sum;
}

//16>> Given an array A[] of size N, construct a Product Array P
// such that P[i] is equal to the product of all the elements
// of A except A[i] // TC o(n) // SC o(n)
vector<long long int> productExceptSelf(vector<long long int>& nums, int n) {
  long long prod = 1;
	int zeroFlag = 0;
	for (int i = 0; i < n; i++) {
		if (nums[i] != 0) prod *= nums[i];
		else zeroFlag += 1;
	}
	if (zeroFlag > 1) {
		for (int i = 0; i < n; i++) nums[i] = 0;
		return nums;
	}
	if (zeroFlag == 1) {
		for (int i = 0; i < n; i++) {
			if (nums[i] != 0) nums[i] = 0;
			else nums[i] = prod;
		}
		return nums;
	}
	for (int i = 0; i < n; i++)
		nums[i] = prod / nums[i];
	return nums;
}

//17>> Given an array of integers, sort the array (in descending order)
// according to count of set bits in binary representation of array
// elements // TC o(nlogn)
static bool comp(int a, int b){
        return __builtin_popcount(a)>__builtin_popcount(b);
    }
    void sortBySetBitCount(int arr[], int n){
        stable_sort(arr,arr+n,comp);
    }

//18>> Minimum Swaps to Sort // TC o(nlogn) // SC(n)
int minSwaps(vector<int>&nums){
	    int n = nums.size();
	    vector<pair<int,int>> v(n);
	    for(int i=0; i<n; i++) v[i] = {nums[i],i};
	    sort(v.begin(),v.end());
	    int c=0;
	    for(int i=0; i<n; i++){
	        if(v[i].second==i) continue;
	        else{
	            c++;
	            swap(v[i],v[v[i].second]);
	            i--;
	        }
	    }
	    return c;
	}

//19>> Find pivot element in a sorted and rotated array
int peakElement(int arr[], int low, int high, int lowerBound, int upperBound){
    int mid = low + (high - low) / 2;
    if (mid == lowerBound) {
        if (mid == upperBound) {
            return mid;
        } else if (arr[mid] >= arr[mid + 1]) {
            return mid;
        }
    } else if (mid == upperBound) {
        if (arr[mid] >= arr[mid - 1]) {
            return mid;
        }
    } else {
        if (arr[mid] >= arr[mid + 1] && arr[mid] >= arr[mid - 1]) {
            return mid;
        } else if (arr[mid] > arr[high]) {
            return peakElement(arr, mid + 1, high, lowerBound, upperBound);
        } else if (arr[mid] < arr[high]) {
            return peakElement(arr, low, mid - 1, lowerBound, upperBound);
        }
    }
    return -1;
}

//20>> Given two sorted arrays arr1 and arr2 of size M and N respectively
// and an element K. The task is to find the element that would be at
// the k’th position of the final sorted array
int kthElement(int arr1[], int arr2[], int n, int m, int k){
        int c1=0;
        int c2=0;
        int x=0;
        while(c1<n && c2<m){
            if(arr1[c1]<=arr2[c2]) {
                x++;
                if(x==k) return arr1[c1];
                c1++;
            }
            else {
                x++;
                if(x==k) return arr2[c2];
                c2++;
            }
        }
        while(c1<n){
            x++;
            if(x==k) return arr1[c1];
            c1++;
        }
        while(c2<m){
            x++;
            if(x==k) return arr2[c2];
            c2++;
        }
    }

    //21>> Weighted Job Scheduling // TC o(nlogn)
    struct Job
{
    int start, finish, profit;
};

// A utility function that is used for sorting events
// according to finish time
bool myfunction(Job s1, Job s2)
{
    return (s1.finish < s2.finish);
}

// A Binary Search based function to find the latest job
// (before current job) that doesn't conflict with current
// job.  "index" is index of the current job.  This function
// returns -1 if all jobs before index conflict with it.
// The array jobs[] is sorted in increasing order of finish
// time.
int binarySearch(Job jobs[], int index) {
    int lo = 0, hi = index - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (jobs[mid].finish <= jobs[index].start) {
            if (jobs[mid + 1].finish <= jobs[index].start)
                lo = mid + 1;
            else return mid;
        }
        else hi = mid - 1;
    }
    return -1;
}
int findMaxProfit(Job arr[], int n) {
    sort(arr, arr+n, myfunction);
    int *table = new int[n];
    table[0] = arr[0].profit;
    for (int i=1; i<n; i++) {
        int inclProf = arr[i].profit;
        int l = binarySearch(arr, i);
        if (l != -1)
            inclProf += table[l];
        table[i] = max(inclProf, table[i-1]);
    }
    int result = table[n-1];
    delete[] table;
    return result;
}

//22>> Given three integers  'A' denoting the first term of an ap
// , 'C' denoting the common difference of an arithmetic sequence and
// an integer 'B'. you need to tell whether 'B' exists in the ap or not
// TC o(1)
int inSequence(int A, int B, int C){
if(A==B)
   return 1;
if(C!=0){
   int res=(B-A)%C;
   if(res==0 && ( ((B-A)<0 && C<0) || ((B-A)>0 && C>0)))
      return 1;
  }
return 0;
}

//23>>  find the smallest number whose factorial contains at least n
// trailing zeroes // TC o(log2 N * log5 N)
bool check(int p, int n){
    int temp = p;
    int count =0;
    int f=5;
    while(f<=p){
        count += temp/f;
        f=f*5;
    }
    return (count>=n);
}
  int findNum(int n){
  if(n==1) return 5;
  int low =0;
  int high =5*n;
  while(low<=high){
      int mid = (low+high)/2;
      if(check(mid,n))
          high=mid-1;
      else low=mid+1;
  }
  return low;
}

//24>> You are given N number of books. Every ith book has Ai number of pages.
//You have to allocate books to M number of students. There can be many ways
// or permutations to do so. In each permutation, one of the M students will
// be allocated the maximum number of pages. Out of all these permutations,
// the task is to find that particular permutation in which the maximum
// number of pages allocated to a student is minimum of those in all the
// other permutations and print this minimum value. Each book will be
// allocated to exactly one student. Each student has to be allocated at
// least one book.Return -1 if a valid assignment is not possible, and
// allotment should be in contiguous order // TC o(nlogn)

bool solve(int *a, int n, int mid, int m){
    int sum =0;
    int stu = 1;
    for(int i=0; i<n; i++){
        if(a[i]>mid) return false;
        if(sum+a[i]>mid){
            stu++;
            sum=a[i];
            if(stu>m) return false;
        }
        else sum+=a[i];
    }
    return true;
}
int findPages(int *a, int n, int m) {
    int lb=0;
    int s=0;
    int ans=0;
    for(int i=0; i<n; i++) s+=a[i];
    int ub=s;
    while(lb<=ub){
        int mid=(lb+ub)/2;
        if(solve(a,n,mid,m)){
            ans=mid;
            ub=mid-1;
        }
        else{
            lb=mid+1;
        }
    }
}


//25>> In place merge sort // TC o(nn)
void merge(int arr[], int start, int mid, int end){
    int start2 = mid + 1;
    // If the direct merge is already sorted
    if (arr[mid] <= arr[start2]) return;
    // Two pointers to maintain start
    // of both arrays to merge
    while (start <= mid && start2 <= end) {
        // If element 1 is in right place
        if (arr[start] <= arr[start2]) start++;
        else {
            int value = arr[start2];
            int index = start2;
            // Shift all the elements between element 1
            // element 2, right by 1.
            while (index != start) {
                arr[index] = arr[index - 1];
                index--;
            }
            arr[start] = value;
            // Update all the pointers
            start++;
            mid++;
            start2++;
        }
    }
}
/* l is for left index and r is right index of the
   sub-array of arr to be sorted */
void mergeSort(int arr[], int l, int r){
    if (l < r) {
        // Same as (l + r) / 2, but avoids overflow
        // for large l and r
        int m = l + (r - l) / 2;
        // Sort first and second halves
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

// ALSO GO THROUGH LIST FOR SPOJ AND SOME OTHER PROBLEMs
