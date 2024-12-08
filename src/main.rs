mod list;
use std::fmt::{Display, Debug};
pub use list::List;

pub struct Node<T> {
    value:T,
    left:Box<Tree<T>>,
    right:Box<Tree<T>>,
}

pub enum Tree<T> {
    Element(Node<T>),
    End,
}


impl <T> Tree <T> 
where T: PartialEq + PartialOrd
{
    pub fn new() -> Self {
        Tree::End
    }
    pub fn create(value:T) -> Self {
        Tree::Element (
            Node {
                value,
                left: Box::new(Tree::End),
                right: Box::new(Tree:: End),
            }
        )
    }
    pub fn append(&mut self, value:T) {
        match self {
            Tree::End => *self = Tree::create(value),
            Tree::Element(ref mut node) => {
                if node.value > value {
                    node.right.append(value)
                } else {
                    node.left.append(value)
                }
            },
        };
    }
}

impl <T> Tree <T>
where T: Debug
{
    pub fn display(&self) {
        match self {
            Tree::End => println!("End!"),
            Tree::Element(node) => {
                node.left.display();
                println!("Value: {:?}", node.value);
                node.right.display();
            },
        };
    }
}

fn main() {
    let mut tree = Tree::new();
    tree.append(10);
    tree.append(5);
    tree.append(15);
    tree.display();


    let mut a = List::new();
    a.append(32);
    a.append(100);
    a.display();
}
