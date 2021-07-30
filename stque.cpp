
// STACK AND queue

//1>> IMPLEMENT queue:
struct QueueNode{
    int data;
    QueueNode *next;
};

class MyQueue {
private:
    int arr[100005];
    int front;
    int rear;

public :
    MyQueue(){front=0;rear=0;}
    void push(int);
    int pop();
};
void MyQueue :: push(int x){
    arr[rear++]=x;
}
int MyQueue :: pop(){
    if(rear==front) return -1;
    else{
     int x = arr[front++];
     return x;
    }
}

//2>> IMPLEMENT STACK:

//3>> Reverse a string using Stack // TC o(n) // SC o(n)
char* reverse(char *s, int len){
    stack<char> k;
    for(int i=0; i<len; i++)
        k.push(s[i]);
    for(int i=0; i<len; i++){
        s[i] = k.top();
        k.pop();
    }
    return s;
}

//4>> Next greater element of an element in the array is the nearest element
// on the right which is greater than the current elemen // TC o(n)
vector<long long> nextLargerElement(vector<long long> a, int n){
    vector <long long> ans;
    stack<long long> s;
    for(int i=n-1;i>=0;i--){
       if(s.empty()) ans.push_back(-1);
       else{
         while(!s.empty() && s.top()<=a[i])
          s.pop();
          if(s.empty())
             ans.push_back(-1);
          else
             ans.push_back(s.top());
        }
       s.push(a[i]);
    }
    reverse(ans.begin(),ans.end());
    return ans;
}

//5>> A celebrity is a person who is known to all but does not know
// anyone at a party, find celibraty // TC o(n) // SC o(n)
int celebrity(vector<vector<int> >& M, int n) {
    if(n==0) return -1;
    stack<int>s;
    for(int i=0;i<n;i++)
        s.push(i);
    while(s.size()>=2){
        int i = s.top();
        s.pop();
        int j = s.top();
        s.pop();
        if(M[i][j]==1) s.push(j);
        else s.push(i);
    }
    if(s.empty()) return -1;
    int ans = s.top();
    s.pop();
    for(int i=0;i<n;i++){
        if(i!=ans){
        if(M[ans][i]==1 || M[i][ans]==0) return -1;
        }
    }
    return ans;
}

// SC o(1) with two pointer approach.

//6>> Evaluation of Postfix Expression // TC o(s) // SC o(s)
int evaluatePostfix(string str){
    stack<int> s;
    for(int i = 0; i < str.size(); i++){
        if(str[i] >= '0' && str[i] <= '9')
            s.push(str[i]-'0');
        else{
            int n1 = s.top();
            s.pop();
            int n2 = s.top();
            s.pop();
            if(str[i] == '*'){
                s.push(n1*n2);
            }
            else if(str[i] == '/'){
                s.push(n2/n1);
            }
            else if(str[i] == '+'){
                s.push(n1+n2);
            }
            else if(str[i] == '-'){
                s.push(n2-n1);
            }
        }
    }
    return s.top();
}

//7>> Implement a method to insert an element at its bottom without using
// any other data structure. // TC o(n)
void insert_at_bottom(char x){
    if(st.isEmpty())
        st.push(x);
    else{
        /* All items are held in Function Call Stack until we
           reach end of the stack. When the stack becomes
           empty, the st.size() becomes 0, the
           above if part is executed and the item is inserted
           at the bottom */
        char a = st.peek();
        st.pop();
        insert_at_bottom(x);
        //push all the items held in Function Call Stack
        //once the item is inserted at the bottom
        st.push(a);
    }
}

//8>> Reverse a stack using recursion // TC o(nn) // SC o(nn)
stack<char> st;
char insert_at_bottom(char x) {
    if(st.size() == 0) st.push(x);
    else{
        char a = st.top();
        st.pop();
        insert_at_bottom(x);
        st.push(a);
    }
}

char reverse() {
    if(st.size()>0) {
        char x = st.top();
        st.pop();
        reverse();
        insert_at_bottom(x);
    }
}

//9>> Sort a Stack using recursion // TC o(nn) // SC o(n)
void sort(stack<int> s){
   stack<int> stk;
    while(!s.empty()){
    int x = s.top();
    s.pop();
    while(!stk.empty() && x < stk.top()){
    s.push(stk.top());
    stk.pop();
    }
    stk.push(x);
    }
    s = stk;
}

//10>> S consisting only of opening and closing parenthesis 'ie '('
//  and ')', find out the length of the longest valid(well-formed)
// parentheses substring // TC o(n) // SC o(1)
int findMaxLen(string s) {
          int n=s.length();
        int r=0,l=0,maxi=0;
        for(int i=0;i<n;i++){
            if(s[i]=='('){
                l++;
            }
            else {
                r++;
            }
            if(l==r){
                maxi=max(maxi,2*r);
            }
            else if(r>l){
                l=r=0;
            }
        }
        l=r=0;
        for(int i=n-1;i>=0;i--){
            if(s[i]=='('){
                l++;
            }
            else {
                r++;
            }
            if(l==r){
                maxi=max(maxi,2*l);
            }
            else if(l>r){
                l=r=0;
            }
        }
        return maxi;
    }

//11>> Maximum Rectangular Area in a Histogram // TC o(n) // SC o(n)
vector<pair<long long, int>> v1;
    vector<pair<long long, int>> v2;
    stack<pair<long long,int>> s;
    void ngr(long long arr[], int n){
        v1.clear();
        for(int i=n-1; i>=0; i--){
            if(s.empty()){
                v1.push_back({0,n});
            }
            else if(s.size()>0 && s.top().first<arr[i]){
                v1.push_back({s.top().first,s.top().second});
            }
            else if(s.size()>0 && s.top().first>= arr[i]){
                while(s.size()>0 && s.top().first>= arr[i]) s.pop();
                if(s.size()==0) v1.push_back({0,n});
                else v1.push_back({s.top().first,s.top().second});
            }
            s.push({arr[i],i});
        }
        reverse(v1.begin(),v1.end());
    }
    void ngl(long long arr[], int n){
        v2.clear();
    while(!s.empty())
       s.pop();
        for(int i=0; i<n; i++){
            if(s.empty()){
                v2.push_back({0,-1});
            }
            else if(s.size()>0 && s.top().first<arr[i]){
                v2.push_back({s.top().first,s.top().second});
            }
            else if(s.size()>0 && s.top().first>= arr[i]){
                while(s.size()>0 && s.top().first>= arr[i]) s.pop();
                if(s.size()==0) v2.push_back({0,-1});
                else v2.push_back({s.top().first,s.top().second});
            }
            s.push({arr[i],i});
        }
    }
    long long getMaxArea(long long arr[], int n){
    long long mx =0;
    ngr(arr,n);
    ngl(arr,n);
    for(int i=0; i<n; i++){
        long long x = arr[i]*(v1[i].second-v2[i].second-1);
        mx = max(mx,x);
    }
    return mx;
}

//12>> Expression contains redundant bracket or not // TC o(n) // SC o(n)
bool checkRedundancy(string& str) {
    stack<char> st;
    for (auto& ch : str) {
        if (ch == ')') {
            char top = st.top();
            st.pop();
            bool flag = true;
            while (top != '(') {
                if (top == '+' || top == '-' ||
                    top == '*' || top == '/')
                    flag = false;
                top = st.top();
                st.pop();
            }
            if (flag == true)
                return true;
        }
        else
            st.push(ch);
    }
    return false;
}

//13>> Implement Stack using Queue // TC o(1)+o(n) // SC o(n)
void QueueStack :: push(int x){
        q1.push(x);
}
int QueueStack :: pop(){
        if (q1.empty())
        return -1;
    int res;
    int s = q1.size();
    while (!q1.empty() && s>1) {
        res = q1.front();
        q1.push(res);
        q1.pop();
        s--;
    }
    res = q1.front();
        q1.pop();

    return res;
}

//14>> Stack Permutations (Check if an array is stack permutation of other)
// TC o(n) // SC o(n)
bool checkStackPermutation(int ip[], int op[], int n) {
    stack <int> tempStack;
    while (!input.empty()) {
        int ele = input.front();
        input.pop();
        if (ele == output.front()) {
            output.pop();
            while (!tempStack.empty()){
                if (tempStack.top() == output.front()){
                    tempStack.pop();
                    output.pop();
                }
                else break;
            }
        }
        else tempStack.push(ele);
    }
    return (input.empty()&&tempStack.empty());
}

//15>> Implement Queue using Stack
void StackQueue :: push(int x){
        s1.push(x);
 }
int StackQueue :: pop(){
         if(s2.empty()){
            if(s1.empty()){
                return -1;
            }
            else{
                while(!s1.empty()){
                    int ele=s1.top();
                    s1.pop();
                    s2.push(ele);
                }
                int ele=s2.top();
                s2.pop();
                return ele;
            }
        }
        else{
            int ele=s2.top();
            s2.pop();
            return ele;
        }
}

//16>> Reverse a Queue using recursion // TC o(n) // system stack o(n)
void solve(queue<int> &q){
if(q.empty()) return;
int data = q.front();
q.pop();
solve(q);
q.push(data);
return;
}
queue<int> rev(queue<int> q){
 solve(q);
 return q;
}

//17>> Reverse the first “K” elements of a queue // TC o(n) //SC o(n)
queue<int> modifyQueue(queue<int> q, int k){
        int i,j;
    queue<int>nq;
    stack<int>st;
    for(i=0;i<k;i++){
        int x=q.front();
        st.push(x);
        q.pop();
    }
    while(!st.empty()){
        int x= st.top();
        nq.push(x);
        st.pop();
    }
    while(!q.empty()){
        int x=q.front();
        nq.push(x);
        q.pop();
    }
    return nq;
}

//18>> Find the first circular tour that visits all Petrol Pumps
// TC o(n) // SC o(1)
int tour(petrolPump p[],int n){
   int def=0,pro=0,start=0;
   for(int i=0; i<n; i++){
       pro += p[i].petrol - p[i].distance;
       if(pro<0){
           start = i+1;
           def+=pro;
           pro =0;

       }
   }
   return (pro+def>=0)? start:-1;
}

//19>> Interleave the first half of the queue with second half
// TC o(n) // SC o(n)
void interLeaveQueue(queue<int>& q) {
    // To check the even number of elements
    if (q.size() % 2 != 0)
        cout << "Input even number of integers." << endl;
    stack<int> s;
    int halfSize = q.size() / 2;
    // Push first half elements into the stack
    // queue:16 17 18 19 20, stack: 15(T) 14 13 12 11
    for (int i = 0; i < halfSize; i++) {
        s.push(q.front());
        q.pop();
    }
    // enqueue back the stack elements
    // queue: 16 17 18 19 20 15 14 13 12 11
    while (!s.empty()) {
        q.push(s.top());
        s.pop();
    }
    // dequeue the first half elements of queue
    // and enqueue them back
    // queue: 15 14 13 12 11 16 17 18 19 20
    for (int i = 0; i < halfSize; i++) {
        q.push(q.front());
        q.pop();
    }
    // Again push the first half elements into the stack
    // queue: 16 17 18 19 20, stack: 11(T) 12 13 14 15
    for (int i = 0; i < halfSize; i++) {
        s.push(q.front());
        q.pop();
    }
    // interleave the elements of queue and stack
    // queue: 11 16 12 17 13 18 14 19 15 20
    while (!s.empty()) {
        q.push(s.top());
        s.pop();
        q.push(q.front());
        q.pop();
    }
}

//20>> Minimum time required to rot all oranges
// TC o(n*m) // SC o(n*m)
int orangesRotting(vector<vector<int>>& grid) {
      queue<pair<pair<int,int>,int>>q;
      for(int i =0 ; i < grid.size();i++){
          for(int j = 0 ;j < grid[0].size();j++){
              if(grid[i][j]==2)
                  q.push({{i,j},0});
          }
      }
      int m = grid.size();
      int n = grid[0].size();
      vector<pair<int,int>> dir;
      dir.push_back({-1,0});
      dir.push_back({1,0});
      dir.push_back({0,-1});
      dir.push_back({0,1});
      int t;
      while(!q.empty()){
          int x= q.front().first.first;
          int y= q.front().first.second;
          t= q.front().second;
          q.pop();
          for(int i =0 ; i < 4 ;i++){
              if(x+dir[i].first<m && x+dir[i].first > -1 && y +dir[i].second<n &&
              y+dir[i].second >-1 && grid[x+dir[i].first][y+dir[i].second]==1){
                  grid[x+dir[i].first][y+dir[i].second]=2;
                  q.push({{x+dir[i].first,y+dir[i].second},t+1});
              }
          }
      }
      for(int i =0 ; i < grid.size();i++){
          for(int j = 0 ;j < grid[0].size();j++){
              if(grid[i][j]==1)
                  return -1;
          }
      }
      return t;
  }

//21>> Distance of nearest cell having 1 in a binary matrix
// TC o(mn) // SC o(nm)
vector<vector<int>> nearest(vector<vector<int>> arr) {
	    int row = arr.size();
	    int col = arr[0].size();
        queue<pair<int,int>>q;
        vector<vector<int>>dist(row,vector<int>(col,INT_MAX));
  for(int i=0;i<arr.size();i++){
      for(int j=0;j<arr[0].size();j++){
          if(arr[i][j]==1){
           q.push({i,j});
           dist[i][j]=0;
          }
      }
  }
  int x_dir[]={-1,0,1,0};
  int y_dir[]={0,-1,0,1};
  while(!q.empty()){
      pair<int,int>temp=q.front();
      int i=temp.first;
      int j=temp.second;
      q.pop();
      for(int k=0;k<4;k++){
          int x_new=i+x_dir[k];
          int y_new=j+y_dir[k];
          if(x_new>=0&&x_new<row&&y_new>=0&&y_new<col&&dist[x_new][y_new]>dist[i][j]+1){
              dist[x_new][y_new]=dist[i][j]+1;
              q.push({x_new,y_new});
          }
      }
  }
  return dist;
}

//22>> First negative integer in every window of size k // TC o(n) // SC o(n)
int n,k;
	  cin >> n;
	  int arr[n];
	  queue <int> neg;
	  for(int i = 0; i < n; ++i)
	      cin >> arr[i];
	  cin >> k;
	  for(int i = 0; i < k; ++i)
	      if(arr[i] < 0) neg.push(arr[i]);
	  for(int i = k; i <= n - 1; ++i){
	      if(neg.size() != 0)
	      cout<<neg.front()<<" ";
	      else cout<<"0 ";
	      if(arr[i-k] < 0) neg.pop();
	      if(arr[i] < 0) neg.push(arr[i]);
	  }
	   if(neg.size() != 0)
	      cout<<neg.front()<<" ";
	      else cout<<"0 ";
	  cout<<"\n";

//23>> Minimum sum of squares of character counts in a given string after
// removing “k” characters. // TC o(nlogn) // SC o(n)
int minValue(string str, int k){
        int l = str.length();
        if (k >= l) return 0;
        map<char,int>m;
        for (int i = 0; i < l; i++){
        m[str[i]]++;
        }
        priority_queue<int> q;
        for(auto i : m ){
        q.push(i.second);
        }
        while (k--) {
        int temp = q.top();
        q.pop();
        temp = temp - 1;
        q.push(temp);
        }
        int result = 0;
        while (!q.empty()) {
        int temp = q.top();
        result += temp * temp;
        q.pop();
        }
        return result;
    }
