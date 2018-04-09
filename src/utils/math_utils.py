def safe_divide(a, b):
    return a/b if b != 0 else 0

if __name__ == "__main__":
    assert safe_divide(1.0, 2) == 0.5
    assert safe_divide(1.0, 0) == 0
    assert safe_divide(0, 2) == 0
