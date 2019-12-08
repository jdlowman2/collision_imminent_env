

class Rectangle:
    def __init__(self, length, width, x, y):
        self.length = length
        self.width = width
        self.x = x
        self.y = y

    def is_inside(self, x, y):
        lower_x = self.x - self.length/2.0
        upper_x = self.x + self.length/2.0
        lower_y = self.y - self.width/2.0
        upper_y = self.y + self.width/2.0

        if x <= upper_x and x >= lower_x:
            if y <= upper_y and y >= lower_y:
                return True

        return False

    def get_width(self):
        return self.width

    def get_length(self):
        return self.length

    def get_left_boundary(self):
        return self.y + self.width/2.0

    def get_right_boundary(self):
        return self.y - self.width/2.0

    def get_start(self):
        return self.x - self.length/2.0

    def get_end(self):
        return self.x + self.length/2.0

    def __repr__(self):
        return "Rectangle: "+ \
                    "x = " + str(self.x) + " | " + \
                    "y = " + str(self.y) + " | " + \
                    "l = " + str(self.length) + " | " + \
                    "w = " + str(self.width)
