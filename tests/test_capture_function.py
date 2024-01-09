from pylon.utils.capture_constraint import capture_constraint

def constraint(x1, x2, x3):
    return (~x1 | x3) & (~x2 | x3)

def test_capture_constraint():
    print(list(capture_constraint(constraint).models()))

test_capture_constraint()
