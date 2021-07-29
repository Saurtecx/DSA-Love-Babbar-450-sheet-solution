// BINARY SEARCH TREE

//1>> Insert a node in a BST // TC o(h)
Node* insert(Node* node, int data){
   if(node==NULL)
   return new Node(data);
if(data<node->data) node->left=insert(node->left,data);
if(node->data<data) node->right=insert(node->right,data);
else return node;
}

//2>> Delete Node in a BST // TC o(h)
TreeNode* deleteNode(TreeNode* root, int key) {
    if(!root)return root;
    if(key<root->val)
        root->left = deleteNode(root->left,key);
    else if(key>root->val)
        root->right = deleteNode(root->right,key);
    else if(root->left && root->right){
        root->val = findmin(root->right)->val;
        root->right = deleteNode(root->right,root->val);
    }
    else root = (root->left) ?root->left : root->right;
    }
    return root;
}
TreeNode* findmin(TreeNode* root){
    if(root){
        while(root->left != nullptr)
            root = root->left;
    }
    return root;
}

//3>> Minimum element in BST
int minValue(Node* node){
    if(!node) return -1;
    while(node->left) node=node->left;
    return node->data;
}

//4>> find the inorder successor and predecessor of a given key
void findPreSuc(Node* root, Node*& pre, Node*& suc, int key){
if(root == NULL) return;
findPreSuc(root->left, pre, suc, key);
if(suc == NULL && root->key < key) pre = root;
if(suc == NULL && root->key > key){
suc = root;
return;
}
findPreSuc(root->right, pre, suc, key);
}

//5>> Check for BST // TC o(n) // SC o(h)
bool checkNode(Node* node,int min,int max){
if(node==NULL) return 1;
if(node->data<min||node->data>max) return 0;
else{
checkNode(node->left,min,node->data-1)&&
checkNode(node->right,node->data+1,max);
   }
}
bool isBST(Node* root){
return checkNode(root,INT_MIN,INT_MAX);
}

//6>> next pointer for every node should be set to point to inorder successor
void inOrder(node* root, node* &prev){
    if(root == NULL) return;
    inOrder(root->left, prev);
    if(prev) prev->next = root;
    prev = root;
    inOrder(root->right, prev);
}
void populateNext(struct node* root){
    node* prev=NULL;
    inOrder(root, prev);
}

//7>> Lowest Common Ancestor in a BST // TC o(n)
Node* LCA(Node *root, int n1, int n2){
  if(root == NULL) return NULL;
  if(root->data > max(n1,n2)) LCA(root->left, n1, n2);
  else if(root->data < min(n1,n2)) LCA(root->right , n1, n2);
  else return root;
}

//8>> BST from preorder // TC o(nn)
TreeNode* bstFromPreorder(vector<int>& preorder) {
        TreeNode* root = NULL;
        for (int &n : preorder) root = insert(root, n);
        return root;
    }
    TreeNode* insert(TreeNode* root, int &n) {
        if (root == NULL)
            return new TreeNode(n);
        if (n > root->val)
            root->right = insert(root->right, n);
        else
            root->left = insert(root->left, n);
        return root;
    }

// TC o(n)
TreeNode * fun(vector<int>&ar,int p,int n,int l,int h){
        if(p==n) return NULL;
        if(ar[p]<h&&ar[p]>l){
            TreeNode *x=new TreeNode(ar[p]);
            x->left=fun(ar,p+1,n,l,ar[p]);
            x->right=fun(ar,p+1,n,ar[p],h);
            return x;
        }
        return fun(ar,p+1,n,l,h);
    }
    TreeNode* bstFromPreorder(vector<int>& preorder) {
        return fun(preorder,0,preorder.size(),INT_MIN,INT_MAX);
    }

//9>> Binary Tree to BST // TC o(nlogn) // SC o(n)
void inorder(vector<int> &in,Node* root){
    if(!root) return;
    inorder(in,root->left);
    in.push_back(root->data);
    inorder(in,root->right);
}
void bst(Node* root, vector<int> &in, int &i){
    if(!root) return;
    bst(root->left,in,i);
    root->data = in[i++];
    bst(root->right,in,i);
}
Node *binaryTreeToBST (Node *root){
vector<int> in;
inorder(in, root);
sort(in.begin(), in.end());
int i = 0;
bst(root,in,i);
return root;
}

//10>> Convert a normal BST to Balanced BST // TC o(n) // SC o(n)
void storeBSTNodes(Node* root, vector<Node*> &nodes) {
    if (root==NULL) return;
    storeBSTNodes(root->left, nodes);
    nodes.push_back(root);
    storeBSTNodes(root->right, nodes);
}
Node* buildTreeUtil(vector<Node*> &nodes, int start,
                   int end){
    if (start > end) return NULL;
    int mid = (start + end)/2;
    Node *root = nodes[mid];
    root->left  = buildTreeUtil(nodes, start, mid-1);
    root->right = buildTreeUtil(nodes, mid+1, end);
    return root;
}
Node* buildTree(Node* root) {
    vector<Node *> nodes;
    storeBSTNodes(root, nodes);
    int n = nodes.size();
    return buildTreeUtil(nodes, 0, n-1);
}

//11>> merge Two bst // TC o(n+m) // SC o(h1 + h2)
void insertNodes(Node *root,stack<Node *> &s){
    while(root){
        s.push(root);
        root=root->left;
    }
}
vector<int> merge(Node *root1, Node *root2){
   stack<Node *> s1;
   stack<Node *> s2;
   vector<int> res;
   // 1.First, push all the elements from root to the left-most leaf node onto a stack. Do this for both trees
   insertNodes(root1,s1);
   insertNodes(root2,s2);
   while(!s1.empty() || !s2.empty()){
       // 2. Peek at the top element of each stack (if non-empty) and print the smaller one.
       // If one the stack empty assign INT_MAX to value coming from that stack
       int a=(!s1.empty() ? s1.top()->data:INT_MAX);
       int b=(!s2.empty() ? s2.top()->data:INT_MAX);
       if(a<=b){
           res.push_back(a);
           Node *temp=s1.top();
           //3. Pop the element off the stack that contains the element we just printed
           s1.pop();
           //4. Add the right child of the element we just popped onto the stack, as well as all its left descendants
           insertNodes(temp->right,s1);
       }
       else{
           res.push_back(b);
           Node *temp=s2.top();
           //3. Pop the element off the stack that contains the element we just printed
           s2.pop();
           //4. Add the right child of the element we just popped onto the stack, as well as all its left descendants
           insertNodes(temp->right,s2);
       }
   }
   return res;
}

//12>> Find Kth largest element in a BST // TC o(h+k) // SC o(1)
void inorder(Node *root,int k,int &n,int &ans){
  if(root==NULL || n>=k) return;
  inorder(root->right,k,n,ans);
  n++;
  if(k==n){
    ans=root->data;
    return;
  }
  inorder(root->left,k,n,ans);
}
int kthLargest(Node *root, int K){
  int n=0;
  int ans;
  inorder(root,K,n,ans);
  return ans;
}

//13>>k-th smallest element in BST // TC o(n) // SC o(n)
void inorder(Node *root,vector<int>&v){
  if(root==NULL)return;
  inorder(root->left,v);
  v.push_back(root->data);
  inorder(root->right,v);
}
int KthSmallestElement(Node *root, int k){
  vector<int>v;
  inorder(root,v);
  return v[k-1];
}

// TC o(n) // SC o(1)
void solve(Node*root,int &kSmall,int &kthSmall){
    if(!root) return;
    solve(root->left,kSmall,kthSmall);
    if(kSmall>=1){
        kthSmall=root->data;
        kSmall--;
    }
    solve(root->right,kSmall,kthSmall);
}
int KthSmallestElement(Node *root, int K){
    int kSmall=K;
    int kthSmall=0;
    solve(root,kSmall,kthSmall);
    if(kSmall!=0) return -1;
    return kthSmall;
}

//14>> Count pairs from 2 BST whose sum is equal to given value "X"
// Naive meathod // TC o(n1h)
For each node value a in BST 1, search the value (x – a) in BST 2
// TC  O(n1 + n2) // SC O(n1 + n2) // two  pointer approach
int countPairs(Node* root1, Node* root2, int x){
    if (root1 == NULL || root2 == NULL) return 0;
    stack<Node*> st1, st2;
    Node* top1, *top2;
    int count = 0;
    while (1){
        while (root1 != NULL) {
            st1.push(root1);
            root1 = root1->left;
        }
        while (root2 != NULL) {
            st2.push(root2);
            root2 = root2->right;
        }
        if (st1.empty() || st2.empty()) break;
        top1 = st1.top();
        top2 = st2.top();
        if ((top1->data + top2->data) == x) {
            count++;
            st1.pop();
            st2.pop();
            root1 = top1->right;
            root2 = top2->left;
        }
        else if ((top1->data + top2->data) < x) {
            st1.pop();
            root1 = top1->right;
        }
        else {
            st2.pop();
            root2 = top2->left;
        }
    }
    return count;
}

//15>> Count BST nodes that lie in a given range
// TC o(n) // SC o(h)
int getCount(Node *root, int l, int h){
  int c=0;
  if(root==NULL) return c;
  if(root->data<=h&&root->data>=l) c++;
  c+=getCount(root->left, l, h);
  c+=getCount(root->right, l, h);
  return c;
}

//16>> Check whether BST contains Dead end // TC o(n) // SC o(n)
bool solve(Node* r, int min_ = 1, int max_ = INT_MAX){
    if(!r) return false;
    if(min_ == max_) return true;
    return solve(r->left, min_, r->data - 1) ||
           solve(r->right, r->data + 1, max_);
}
bool isDeadEnd(Node *root){
    return solve(root);
}

//17>> Largest BST in a Binary Tree // TC o(n) // SC o(n)
class bst{
public:
  bool isbst;
  int size;
  int leftmax;
  int rightmin;
};

bst solve(Node*root){
   bst p;
   if(root==NULL){
     p.isbst=true;
     p.size=0;
     p.leftmax=INT_MIN;
     p.rightmin=INT_MAX;
   return p;
}
   bst left=solve(root->left);
   bst right=solve(root->right);

   if(root->data > left.leftmax && root->data < right.rightmin
     && left.isbst && right.isbst){
     p.isbst=true;
     p.size=left.size+right.size+1;
     p.leftmax=max(left.leftmax,max(right.leftmax,root->data));
     p.rightmin=min(right.rightmin,min(left.rightmin,root->data));
 }
   else{
     p.isbst=false;
     p.size=max(left.size,right.size);
 }
   return p;
}
int largestBst(Node *root){
  bst p=helper(root);
  return p.size;
}

//18>> Flatten BST to sorted list // TC o(n) // SC o(h)
void inorder(node* curr, node*& prev) {
    if (curr == NULL) return;
    inorder(curr->left, prev);
    prev->left = NULL;
    prev->right = curr;
    prev = curr;
    inorder(curr->right, prev);
}
node* flatten(node* root) {
    node* dummy = new node(-1);
    node* prev = dummy;
    inorder(root, prev);
    prev->left = NULL;
    prev->right = NULL;
    node* ret = dummy->right;
    delete dummy;
    return ret;
}

//19>> Find the median of BST in O(n) time and O(1) space
/* Function to count nodes in a  binary search tree
   using Morris Inorder traversal*/
int counNodes(struct Node *root) {
    struct Node *current, *pre;
    // Initialise count of nodes as 0
    int count = 0;
    if (root == NULL) return count;
    current = root;
    while (current != NULL) {
        if (current->left == NULL) {
            // Count node if its left is NULL
            count++;
            // Move to its right
            current = current->right;
        }
        else{
            /* Find the inorder predecessor of current */
            pre = current->left;
            while (pre->right != NULL && pre->right != current)
                pre = pre->right;
            /* Make current as right child of its
               inorder predecessor */
            if(pre->right == NULL){
                pre->right = current;
                current = current->left;
            }
            /* Revert the changes made in if part to
               restore the original tree i.e., fix
               the right child of predecssor */
            else{
                pre->right = NULL;
                // Increment count if the current
                // node is to be visited
                count++;
                current = current->right;
            } /* End of if condition pre->right == NULL */
        } /* End of if condition current->left == NULL*/
    } /* End of while */
    return count;
}
/* Function to find median in O(n) time and O(1) space
   using Morris Inorder traversal*/
int findMedian(struct Node *root) {
   if (root == NULL) return 0;
    int count = counNodes(root);
    int currCount = 0;
    struct Node *current = root, *pre, *prev;
    while (current != NULL) {
        if (current->left == NULL){
            // count current node
            currCount++;
            // check if current node is the median
            // Odd case
            if (count % 2 != 0 && currCount == (count+1)/2)
                return prev->data;
            // Even case
            else if (count % 2 == 0 && currCount == (count/2)+1)
                return (prev->data + current->data)/2;
            // Update prev for even no. of nodes
            prev = current;
            //Move to the right
            current = current->right;
        }
        else{
            /* Find the inorder predecessor of current */
            pre = current->left;
            while (pre->right != NULL && pre->right != current)
                pre = pre->right;
            /* Make current as right child of its inorder predecessor */
            if (pre->right == NULL){
                pre->right = current;
                current = current->left;
            }
            /* Revert the changes made in if part to restore the original
              tree i.e., fix the right child of predecssor */
            else{
                pre->right = NULL;
                prev = pre;
                // Count current node
                currCount++;
                // Check if the current node is the median
                if (count % 2 != 0 && currCount == (count+1)/2 )
                    return current->data;
                else if (count%2==0 && currCount == (count/2)+1)
                    return (prev->data+current->data)/2;
                // update prev node for the case of even
                // no. of nodes
                prev = current;
                current = current->right;
            } /* End of if condition pre->right == NULL */
        } /* End of if condition current->left == NULL*/
    } /* End of while */
}

//20>> Replace every element with the least greater element on its right
//TC o(nn)
insert in bst replace with inorder successor if not found then -1;

// TC o(n)
void printNGE(int arr[], int n) {
stack < int > s;
/* push the first element to stack */
s.push(arr[0]);
// iterate for rest of the elements
for (int i = 1; i < n; i++) {
	if (s.empty()) {
	s.push(arr[i]);
	continue;
	}
	/* if stack is not empty, then
	pop an element from stack.
	If the popped element is smaller
	than next, then
	a) print the pair
	b) keep popping while elements are
	smaller and stack is not empty */
	while (s.empty() == false && s.top() < arr[i]) {
		cout << s.top() << " --> " << arr[i] << endl;
		s.pop();
	}
	/* push next to stack so that we can find
	next greater for it */
	s.push(arr[i]);
}
/* After iterating over the loop, the remaining
elements in stack do not have the next greater
element, so print -1 for them */
while (s.empty() == false) {
	cout << s.top() << " --> " << -1 << endl;
	s.pop();
}
}

//21>> Given "n" appointments, find the conflicting appointments
Check one by one process all appointments from the second appointment
to last. For every appointment i, check if it conflicts with
i-1, i-2, … 0. The time complexity of this method is O(n2).

// TC o(nlogn)
1) Create an Interval Tree, initially with the first appointment.
2) Do following for all other appointments starting from the second one.
   a) Check if the current appointment conflicts with any of the existing
     appointments in Interval Tree.  If conflicts, then print the current
     appointment.  This step can be done O(Logn) time.
   b) Insert the current appointment in Interval Tree. This step also can
      be done O(Logn) time.
check the code in gfg.
