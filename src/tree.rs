use std::fmt::Debug;

pub struct Node<T> {
    value: T,
    left: Box<Tree<T>>,
    right: Box<Tree<T>>,
}

pub enum Tree<T> {
    Element(Node<T>),
    End,
}

impl<T> Tree<T>
where
    T: PartialEq + PartialOrd,
{
    pub fn new() -> Self {
        Tree::End
    }
    pub fn create(value: T) -> Self {
        Tree::Element(Node {
            value,
            left: Box::new(Tree::End),
            right: Box::new(Tree::End),
        })
    }
}
impl<T> Tree<T>
where
    T: PartialEq + PartialOrd,
{
    pub fn append(&mut self, value: T) {
        let mut current = self;
        loop {
            match current {
                Tree::End => {
                    *current = Tree::create(value);
                    break;
                }
                Tree::Element(ref mut node) => {
                    if value < node.value {
                        current = &mut node.left;
                    } else {
                        current = &mut node.right;
                    }
                }
            }
        }
    }
}

impl<T> Tree<T>
where
    T: Debug,
{
    pub fn display(&self) {
        match self {
            Tree::End => println!("End!"),
            Tree::Element(node) => {
                node.left.display();
                println!("Value: {:?}", node.value);
                node.right.display();
            }
        };
    }
}

fn main() {
    let mut tree = Tree::new();
    tree.append(10);
    tree.append(5);
    tree.append(15);
    tree.append(1);
    tree.append(100);
    tree.display();

    // let mut a = List::new();
    // a.append(32);
    // a.append(100);
    // a.display();
}
