mod list;
mod tree;
pub use list::List;
pub use tree::Tree;


fn main() {
    let mut tree = Tree::new();
    tree.append(10);
    tree.append(5);
    tree.append(15);
    tree.append(1);
    tree.append(100);
    tree.display();


    let mut a = List::new();
    a.append(32);
    a.append(100);
    a.display();
}
