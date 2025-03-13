Let B be an n x n bidiagonal matrix with diagonal entries
s[1], . . . , s[n]
and superdiagonal entries
e[1] , . . . , e[n-1]
# Theres one less superdiagonal

old_cosine = 1
f = s[1]
g = e[1]
# Is this inclusive, is this zero-based indexing?
for i = 1 to n − 1:
    call ROT ( f , g, cosine, sine, r)
    if ( i!= 1) {
        e[i-1] = oldsine*r
    }
    f = old_cosine * r
    g = s[i + 1] * sine
    h = s[i + 1] * cosine
    call ROT (f, g, cosine, sine, r)
# Store the result into the input matrix
    s[i] = r
    f = h
# This won't exist if we iterate to the second to last row as they only go to e[n-1]
    g = e[i + 1]
    old_cosine = cosine
    old_sine = sine
end for
# `h` is not set as a default value and neither is sine so this would error if we didn't enter the loop
# additionally lets assume it's a value and sine is constant, how would that adapt to all 2x2 matrices??
e[n-1] = h * sine
s[n] = h * cosine


















ROT (f, g, cosine, sine, r) :: Takes (f, g) as input and returns (r, cosine, sine)


if ( f = 0) then
    cosine = 0;
    sine = 1;
    r = g;
elseif ( f.abs() > g.abs() ) then
    t = g/f;
    tt = square_root(1 + t^2);
    cosine = 1 / tt;
    sine = t * cosine;
    r = f * tt;
else
    t = f/g;
    tt = square_root(1 + t^2);
    sine = 1 / tt;
    cosine = t * sine;
    r = g * tt;
endif
return (r, sine, cosine)
