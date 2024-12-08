mod list;
use std::fmt::{Display, Debug};
pub use list::List;

fn main() {
    let mut a = List::new();
    a.append(32);
    a.append(100);
    a.display();
}
