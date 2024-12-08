use std::fmt::Debug;

pub struct Node<T> {
    value: T,
    next: Box<List<T>>,
}

pub enum List<T> {
    Element(Node<T>),
    End,
}

impl<T> List<T> {
    pub fn new() -> Self {
        List::End
    }
    pub fn create(value: T) -> Self {
        List::Element(Node {
            value,
            next: Box::new(List::End),
        })
    }
    pub fn append(&mut self, value: T) {
        match self {
            List::End => {
                *self = List::create(value);
            }
            List::Element(ref mut node) => {
                let mut current = &mut node.next;
                while let List::Element(ref mut next_node) = **current {
                    current = &mut next_node.next;
                }
                *current = Box::new(List::create(value));
            }
        }
    }
}

impl<T> List<T>
where
    T: Debug,
{
    pub fn display(&self) {
        match self {
            List::End => println!("End!"),
            List::Element(node) => {
                println!("Node: {:?}", node.value);
                node.next.display();
            }
        }
    }
}
