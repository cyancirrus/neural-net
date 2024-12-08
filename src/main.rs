use std::fmt::{Display, Debug};

struct Node<T> {
    value:T,
    next:Box<List<T>>,
}

enum List<T> {
    Node(Node<T>),
    End,
}

impl <T> List <T> 
where T : Debug 
{
    pub fn new(value:T) -> Self {
        List::Node(
            Node {
                value,
                next: Box::new(List::End),
            }
        )
    }
    pub fn append(&mut self, value:T) {
        match self {
            List::End => {
                *self = List::new(value);
            }
            List :: Node(ref mut node) => {
                let mut current = &mut node.next;
                while let List::Node(ref mut next_node) = **current {
                    current = &mut next_node.next;
                }
                *current = Box::new(List::new(value));
            }
        }
    }
    pub fn display(&self) {
        match self {
            List::End => println!("End!"),
            List::Node(node) => {
                println!("Node: {:?}", node.value);
                node.next.display();
            }
        }
    }
}

fn main() {
    let mut a = List::new(32);
    a.append(100);
    a.display();
}
