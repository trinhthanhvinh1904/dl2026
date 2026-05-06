def function(x):
    return x ** 2

def derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

def gradient_descent(f, x0, l, min=1e-6):
    x = x0
    step = 0
    print(f"{'Step'} | {'x'} | {'f(x)'} | {'grad'}")
    while True:
        fx_val = f(x)
        grad = derivative(f, x)
        print(f"{step} | {x} | {fx_val} | {grad}")
        if abs(grad) < min:
            break
        x = x - l * grad
        step += 1
    return x

result = gradient_descent(f=function,x0=10,l=1)
print("\nResult:", result)



    
    
        