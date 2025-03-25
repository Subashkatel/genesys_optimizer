import math

def get_factors(n) -> list:
    """Get all factors of a number.
    Args:
        n (int): The number to get factors for.
    Returns:
        list: A sorted list of factors of the numbers.
    """
    
    if n <= 0:
        raise ValueError("Input must be a positive integer.")
    
    factors = []

    # check up to sqrt(n) and include both factors
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            factors.append(i)
            if i != n // i: # avoid adding the square root twice.
                factors.append(n // i)
    return sorted(factors)