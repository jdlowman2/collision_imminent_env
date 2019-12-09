

class Rectangle:
    def __init__(self, length, width, x, y):
        self.length = length
        self.width = width
        self.x = x
        self.y = y

    def is_inside(self, x, y):
        # Checks if the point x,y is within the rectangle
        lower_x = self.x - self.length/2.0
        upper_x = self.x + self.length/2.0
        lower_y = self.y - self.width/2.0
        upper_y = self.y + self.width/2.0

        if lower_x <= x <= upper_x:
            # if y <= upper_y and y >= lower_y:
            if lower_y <= y <= upper_y:
                return True

        return False

    def get_corners(self):
        corners = []

        for delta_x in [self.length/2.0, -self.length/2.0]:
            for delta_y in [self.width/2.0, -self.width/2.0]:
                corners.append([self.x + delta_x, self.y + delta_y])

        return corners

    def intersects(self, other):
        if self.is_inside(other.x, other.y) or other.is_inside(self.x, self.y):
            return True

        for c in self.get_corners():
            if other.is_inside(c[0], c[1]):
                return True

        for c in other.get_corners():
            if self.is_inside(c[0], c[1]):
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
