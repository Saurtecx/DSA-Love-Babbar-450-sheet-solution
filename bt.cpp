// BINARY TREE

//1>> Level order traversal // TC o(n) // SC o(n)
vector<int> levelOrder(Node* node){
  vector<int> arr;
if(node==NULL) return arr;
queue<Node *> q;
q.push(node);
  while(!q.empty()){
  struct Node* temp = q.front();
  arr.push_back(temp->data);
  q.pop();
  if(temp->left!=NULL) q.push(temp->left);
  if(temp->right!=NULL)q.push(temp->right);
  }
return arr;
}

//2>> Reverse Level Order Traversal // TC o(n) // SC o(n)
vector<int> reverseLevelOrder(Node *root){
    vector<int> ans;
   if(!root) return ans;
   Node *temp = root;
   queue<Node *> q;
   q.push(root);
   while(!q.empty()){
       temp = q.front();
       ans.push_back(temp->data);
       if(temp->right) q.push(temp->right);
       if(temp->left) q.push(temp->left);
       q.pop();
   }
   reverse(ans.begin(),ans.end());
   return ans;
}

//3>> Height of Binary Tree // TC o(n) // SC o(n)
int height(struct Node* node){
        if(!node) return 0;
        int x = height(node->left);
        int y = height(node->right);
        return (max(x,y)+1);
    }

//4>> Diameter of a tree // TC o(n) // SC o(h)
int ma;
int func(Node *root){
    if(!root) return 0;
    int x = func(root->left);
    int y = func(root->right);
    ma = max(ma,x+y+1);
    return (max(x,y)+1);
}
int diameter(Node* root) {
    ma = INT_MIN;
    int x = func(root);
    return ma;
}

//5>> Mirror of a tree // TC o(n) // SC o(h)
void mirror(Node* node){
    if(!node) return;
    mirror(node->left);
    mirror(node->right);
    swap(node->left,node->right);
}

//6>> Inorder traversal
void inorder(Node *root){
    if (root == nullptr) return;
    inorder(root->left);
    cout << root->data << " ";
    inorder(root->right);
}
// iterative
void inorderIterative(Node *root){
    stack<Node*> stack;
    Node *curr = root;
    while (!stack.empty() || curr != nullptr){
        if (curr != nullptr){
            stack.push(curr);
            curr = curr->left;
        }
        else{
            curr = stack.top();
            stack.pop();
            cout << curr->data << " ";
            curr = curr->right;
        }
    }
}

//7>> preorder traversal
void preorder(Node *root){
    if (root == nullptr) return;
    cout << root->data << " ";
    preorder(root->left);
    preorder(root->right);
}
//iterative
void preorderIterative(Node *root){
    if (root == nullptr) return;
    stack<Node*> stack;
    stack.push(root);
    while (!stack.empty()){
        Node *curr = stack.top();
        stack.pop();
        cout << curr->data << " ";
        if (curr->right)
            stack.push(curr->right);
        if (curr->left)
            stack.push(curr->left);
    }
}

//8>> postorder traversal
void postorder(Node *root){
    if (root == nullptr) return;
    postorder(root->left);
    postorder(root->right);
    cout << root->data << " ";
}
// iterative
void postorderIterative(Node* root){
    stack<Node*> stk;
    stk.push(root);
    stack<int> out;
    while (!stk.empty()){
        Node *curr = stk.top();
        stk.pop();
        out.push(curr->data);
        if (curr->left)
            stk.push(curr->left);
        if (curr->right)
            stk.push(curr->right);
    }
    while (!out.empty()){
        cout << out.top() << " ";
        out.pop();
    }
}

//9>> Left View of a tree // TC o(n) // SC o(h)
vector<int> leftView(Node *root){
   vector<int> ans;
   queue<Node *> q;
   if(!root) return ans;
   q.push(root);
   while(!q.empty()){
       int sz = q.size();
       ans.push_back(q.front()->data);
       while(sz--){
           Node *t = q.front();
           q.pop();
           if(t->left) q.push(t->left);
           if(t->right) q.push(t->right);
       }
   }
   return ans;
}

//10>> Right View of Tree // TC o(n) // SC o(h)
vector<int> rightView(Node *root){
   vector<int> ans;
   queue<Node *> q;
   if(!root) return ans;
   q.push(root);
   while(!q.empty()){
       int sz = q.size();
       Node *t;
       while(sz--){
           t = q.front();
           q.pop();
           if(t->left) q.push(t->left);
           if(t->right) q.push(t->right);
       }
       ans.push_back(t->data);
   }
   return ans;
}

//11>> Top View of a tree // TC o(n) // SC o(n)
void topView(struct Node *root){
    map<int,int> m;
    queue<pair<Node*,int>> q;
    if(!root) return;
    q.push({root,0});
    while(!q.empty()){
        Node * t =q.front().first;
        int h = q.front().second;
        q.pop();
        if(!m[h]) m[h] = t->data;
        if(t->left) q.push({t->left,h-1});
        if(t->right) q.push({t->right,h+1});

    }
    for(auto x:m) cout<<x.second<<" ";
}

//12>> Bottom View of a tree // TC o(n) // SC o(n)
vector <int> bottomView(Node *root){
    map<int,int> m;
    queue<pair<Node*,int>> q;
    vector<int> v;
    if(!root) return v;
    q.push({root,0});
    while(!q.empty()){
        Node * t =q.front().first;
        int h = q.front().second;
        q.pop();
        m[h] = t->data;
        if(t->left) q.push({t->left,h-1});
        if(t->right) q.push({t->right,h+1});

    }
    for(auto x:m) v.push_back(x.second);
    return v;
}

//13>> Zig-Zag traversal of a binary tree // TC o(n) // SC o(n)
vector <int> zigZagTraversal(Node* root){
	vector<int> ans;
   queue<Node *> q;
   if(!root) return ans;
   q.push(root);
   int f=1;
   while(!q.empty()){
       int sz = q.size();
       vector<int> temp;
       while(sz--){
           Node *t = q.front();
           q.pop();
           temp.push_back(t->data);
           if(t->left) q.push(t->left);
           if(t->right) q.push(t->right);
       }
       if(f%2==0) reverse(temp.begin(),temp.end());
       for(int i=0; i<temp.size(); i++) ans.push_back(temp[i]);
       f=!f;
   }
   return ans;
}

//14>> Check if a tree is balanced or not // TC o(n) // SC o(h)
int f=0;
int solve(Node *root){
    if(!root) return 0;
    int x = solve(root->left);
    int y = solve(root->right);
    if(abs(x-y)>1) return f=0;
    return max(x,y)+1;
}
bool isBalanced(Node *root){
    f=1;
    solve(root);
    return f;
}

//15>> Diagonal Traversal of Binary Tree // TC o(n) // SC o(n)
vector<int> diagonal(Node *root){
   vector<int> v;
   queue<Node*> q;
   if(!root) return v;
   q.push(root);
   while(!q.empty()){
       Node * t=q.front();
       q.pop();
       while(t){
           if(t->left) q.push(t->left);
           v.push_back(t->data);
           t=t->right;
       }
   }
   return v;
}

//16>> Boundary traversal of a Binary tree
void leftTree(Node* root,vector<int> &ans){
    if(!root) return;
    if(root->left){
        ans.push_back(root->data);
        leftTree(root->left,ans);
    }
    else if(root->right){
        ans.push_back(root->data);
        leftTree(root->right,ans);
    }
}
void leaf(Node* root, vector<int> &ans){
    if(!root) return;
    leaf(root->left,ans);
    if(!root->left && !root->right) ans.push_back(root->data);
    leaf(root->right,ans);
}
void rightTree(Node *root, vector<int> &ans){
    if(!root) return;
    if(root->right){
        rightTree(root->right,ans);
        ans.push_back(root->data);
    }
    else if(root->left){
        rightTree(root->left,ans);
        ans.push_back(root->data);
    }
}
vector <int> printBoundary(Node *root){
     vector<int> ans;
     if(!root) return ans;
     ans.push_back(root->data);
     leftTree(root->left,ans);
     leaf(root,ans);
     rightTree(root->right,ans);
     return ans;
}

//17>> Binary Tree to DLL // TC o(n) // SC o(h)
void solve(Node *root, Node* &head, Node* &prev, int &f){
    if(!root) return;
    solve(root->left,head,prev,f);
    if(f==0){
        head=root;
        prev=root;
        f=1;
    }
    else{
        prev->right=root;
        prev->right->left=prev;
        prev=prev->right;
    }
    solve(root->right,head,prev,f);
}

Node * bToDLL(Node *root){
    Node *prev =NULL;
    Node *head=NULL;
    int f=0;
    solve(root,head,prev,f);
    return head;
}

//18>> Convert Binary tree into Sum tree // TC o(n) // SC o(h)
int solve(Node *root){
    if(!root) return 0;
    int a = solve(root->left);
    int b = solve(root->right);
    int x = root->data;
    root->data=a+b;
    return (a+b+x);
}
void toSumTree(Node *node){
    solve(node);
}

//19>> Construct Tree from Inorder & Preorder // TC o(n) // SC o(n)
int idx =0;
unordered_map<int,int> m;
Node* solve(int in[], int pre[], int lb, int ub){
    if(lb>ub) return NULL;
    Node* res = new Node(pre[idx++]);
    if(lb==ub) return res;
    int mid = m[res->data];
    res->left = solve(in,pre,lb,mid-1);
    res->right = solve(in,pre,mid+1,ub);
    return res;
}

Node* buildTree(int in[],int pre[], int n){
idx=0;
m.clear();
for(int i=0; i<n; i++) m[in[i]] = i;
return solve(in,pre,0,n-1);
}

//20>> Check whether it is a Sum Tree or not. // TC o(n) // SC o(h)
int f=1;
int solve(Node *root){
    if(!root) return 0;
    if(!root->left && !root->right) return root->data;
    if(f==0) return 0;
    int a = solve(root->left);
    int b = solve(root->right);
    int x = root->data;
    if(a+b!=x) f=0;
    return (a+b+x);
}
bool isSumTree(Node* root){
    f=1;
    solve(root);
    return f;
}

//21>> Minimum swap required to convert binary tree to binary search tree
// TC o(nlogn) // SC o(n)
int minSwaps(vector<int> &v){
    vector<pair<int,int> > t(v.size());
    int ans = 0;
    for(int i = 0; i < v.size(); i++)
        t[i].first = v[i], t[i].second = i;
    sort(t.begin(), t.end());
    for(int i = 0; i < t.size(); i++){
        if(i == t[i].second)
            continue;
        else{
            swap(t[i].first, t[t[i].second].first);
            swap(t[i].second, t[t[i].second].second);
        }
        if(i != t[i].second)
            --i;
        ans++;
    }
    return ans;
}

//22>> Check if all leaf nodes are at same level or not
// TC o(n) // SC o(h)
int ans = 1;
void solve(Node* root,int h, int &ma){
    if(!root) return;
    if(ans==0) return;
    if(!root->left && !root->right){
        if(ma==-1) ma=h;
        else{
            if(ma!=h) ans=0;
        }
        return;
    }
    solve(root->left,h+1, ma);
    solve(root->right,h+1, ma);
}
bool check(Node *root){
    ans=1;
    int ma = -1;
    int h=0;
    solve(root,h,ma);
    return ans;
}

//23>> Check if a Binary Tree contains duplicate subtrees of size
// 2 or more // TC o(n) // SC o(n)
unordered_map<string,int> m;
string solve(Node* root){
    if(!root) return "$";
    string s ="";
    if(!root->left && !root->right){
        s = to_string(root->data);
        return s;
    }
    s += to_string(root->data);
    s+= solve(root->left);
    s+= solve(root->right);
    m[s]++;
    return s;
}
bool dupSub(Node *root){
     m.clear();
     solve(root);
     for(auto x:m)
         if(x.second>=2) return true;
     return false;
}

//24>> Sum of nodes on the longest path from root to leaf node
// TC o(n)
vector<int> solve(Node* root){
    if(!root) return {0,0};
    vector<int> a = solve(root->left);
    vector<int> b = solve(root->right);
    if(a[0]>b[0]) return {a[0]+1,a[1]+root->data};
    else if(a[0]<b[0]) return {b[0]+1,b[1]+root->data};
    else return {a[0]+1,max(a[1],b[1])+root->data};
}
int sumOfLongRootToLeafPath(Node* root){
	vector<int> v = solve(root);
	return v[1];
}

//25>> Find largest subtree sum in a tree // TC o(n) // SC o(n)
int findLargestSubtreeSumUtil(Node* root, int& ans) {
    if (root == NULL) return 0;
    int currSum = root->key +
      findLargestSubtreeSumUtil(root->left, ans)
      + findLargestSubtreeSumUtil(root->right, ans);
    ans = max(ans, currSum);
    return currSum;
}
int findLargestSubtreeSum(Node* root) {
    if (root == NULL) return 0;
    int ans = INT_MIN;
    findLargestSubtreeSumUtil(root, ans);
    return ans;
}

//26>> Maximum Sum of nodes in Binary tree such that no two are adjacent
pair<int, int> maxSumHelper(Node *root){
    if (root==NULL){
        pair<int, int> sum(0, 0);
        return sum;
    }
    pair<int, int> sum1 = maxSumHelper(root->left);
    pair<int, int> sum2 = maxSumHelper(root->right);
    pair<int, int> sum;
    sum.first = sum1.second + sum2.second + root->data;
    sum.second = max(sum1.first, sum1.second) +
                 max(sum2.first, sum2.second);
    return sum;
}

int maxSum(Node *root){
    pair<int, int> res = maxSumHelper(root);
    return max(res.first, res.second);
}

//map approach
unordered_map<Node*,int> dp;
int func(Node *root){
  if(!root) return 0;
  if(dp[root]) return dp[root];

  int inc = root->data;
  if(root->left){
    inc += func(root->left->left);
    inc += func(root->lrft->right);
  }
  if(root->right){
    inc += func(root->right->left);
    inc += func(root->right->left);
  }
  int exc = func(root->left) + func(root->right);
  dp[root] = max(inc,exc);
  return dp[root];
}

//27>> Print all "K" Sum paths in a Binary tree
void solve(Node *root, vector<int>& path, int k) {
    if (!root) return;
    path.push_back(root->data);
    solve(root->left, path, k);
    solve(root->right, path, k);
    int f = 0;
    for (int j=path.size()-1; j>=0; j--){
        f += path[j];
        if (f == k)
            printVector(path, j);
    }
    path.pop_back();
}
void printKPath(Node *root, int k) {
    vector<int> path;
    solve(root, path, k);
}

//28>> Lowest Common Ancestor in a Binary Tree // TC o(n) // SC o(h)
Node* lca(Node* root ,int n1 ,int n2 ){
   if(!root) return NULL;
   if(root->data==n1 || root->data==n2) return root;
   Node* l = lca(root->left,n1,n2);
   Node* r = lca(root->right,n1,n2);
   if(l and r) return root;
   if(l) return l;
   else return r;
}

//29>> Min distance between two given nodes of a Binary Tree
// TC o(n) // SC o(h)
Node* lca(Node* root ,int n1 ,int n2 ){
   if(!root) return NULL;
   if(root->data==n1 || root->data==n2) return root;
   Node* l = lca(root->left,n1,n2);
   Node* r = lca(root->right,n1,n2);
   if(l and r) return root;
   if(l) return l;
   else return r;
}
int solve(Node* root, int val){
    if(!root) return 0;
    if(root->data == val) return 1;
    int a = solve(root->left,val);
    int b = solve(root->right,val);
    if(!a and !b) return 0;
    else return a+b+1;
}
int findDist(Node* root, int a, int b) {
    Node* LCA = lca(root,a,b);
    int x = solve(LCA,a);
    int y = solve(LCA,b);
    return x+y-2;
}

//30>> Check if Tree is Isomorphic // TC O(min(M,N)) // SC O(min(H1,H2))
bool isIsomorphic(Node *root1,Node *root2){
  if(!root1 && !root2) return true;
  if(!root1 || !root2) return false;
  if(root1->data != root2->data) return false;
  bool a = isIsomorphic(root1->left,root2->left) &&
           isIsomorphic(root1->right,root2->right);
  bool b = isIsomorphic(root1->left,root2->right) &&
           isIsomorphic(root1->right,root2->left);
  return (a || b);
}

//31>> Kth ancestor of a node in binary tree // TC o(n)
Node* kthAncestorDFS(Node *root, int node , int &k) {
    if (!root) return NULL;
    if (root->data == node||
       (temp =  kthAncestorDFS(root->left,node,k)) ||
       (temp =  kthAncestorDFS(root->right,node,k))) {
        if (k > 0) k--;
        else if (k == 0) {
            cout<<"Kth ancestor is: "<<root->data;
            return NULL;
        }
        return root;
    }
}

//32>> Construct Binary Tree from String with bracket representation
// TC o(nn) // SC o(n)
int findIndex(string str, int si, int ei){
    if (si > ei) return -1;
    stack<char> s;
    for (int i = si; i <= ei; i++) {
        if (str[i] == '(')
            s.push(str[i]);
        else if (str[i] == ')') {
            if (s.top() == '(') {
                s.pop();
                if (s.empty())
                    return i;
            }
        }
    }
    return -1;
}
Node* treeFromString(string str, int si, int ei){
    if (si > ei) return NULL;
    Node* root = newNode(str[si] - '0');
    int index = -1;
    if (si + 1 <= ei && str[si + 1] == '(')
        index = findIndex(str, si + 1, ei);
    if (index != -1) {
        root->left = treeFromString(str, si + 2, index - 1);
        root->right= treeFromString(str, index + 2, ei - 1);
    }
    return root;
}

//33>> Find all Duplicate subtrees in a Binary tree // TC o(n) // SC o(n)
void inOrder(Node* root,vector<int> &v){
    if(root==NULL) return;
    inOrder(root->left,v);
    v.push_back(root->data);
    inOrder(root->right,v);
}
void traverse(Node* root,map<vector<int>,int> &mp){
    if(root==NULL) return;
    vector<int> v;
    inOrder(root,v);
    sort(v.begin(),v.end());
    mp[v]++;
    traverse(root->left,mp);
    traverse(root->right,mp);
}
void printAllDups(Node* root){
    map<vector<int>,int> mp;
    traverse(root,mp);
    int flag=0;
    for(auto i=mp.begin();i!=mp.end();i++){
        if(i->second>1){
            vector<int> v=i->first;
            cout<<v[0]<<" ";
            flag=1;
        }
    }
    if(flag==0)
     cout<<"-1"<<" ";
}

//34>> Check Mirror in N-ary tree // TC o(n) // SC o(n)
int main(){
int n,k,l,r;
cin>>n>>k;
vector<stack<int> >s(16);
for(int i=0;i<k;i++) {
cin>>l>>r;
s[l].push(r);
}
int f=0;
for(int i=0;i<k;i++){
cin>>l>>r;
if(!s[l].empty()){
if(s[l].top()==r) s[l].pop();
else f=1;
}
else f=1;
}
if(f==1) cout<<"0\n";
else cout<<"1\n";
for(int i=1;i<=15;i++){
while(!s[i].empty())
s[i].pop();
}
return 0;
}
