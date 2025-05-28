class Dict2Class:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

# Example usage
if __name__ == "__main__":
    sample_dict = {'name': 'John', 'age': 30, 'city': 'New York'}
    obj = Dict2Class(sample_dict)
    print(obj)
    print(obj.name)