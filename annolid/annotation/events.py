class Event:
    """
    Represents an event that occurs at a specific time during a video.
    """

    def __init__(self, name, start_time,
                 end_time=None, color=None,
                 description=None, notes=None,
                 location=None,
                 participants=None):
        """
        Initializes an Event object with the specified properties.

        Args:
            name (str): The name of the event.
            start_time (float): The time when the event starts.
            end_time (float, optional): The time when the event ends.
            color (str, optional): The color associated with the event.
            description (str, optional): A description of the event.
            notes (str, optional): Any additional notes or comments about the event.
            location (str, optional): The location of the event.
            participants (List[str], optional): A list of individuals or groups involved in the event.
        """
        self.name = name
        self.start_time = start_time
        self.end_time = end_time
        self.color = color
        self.description = description
        self.notes = notes
        self.location = location
        self.participants = participants

    def __repr__(self):
        """
        Returns a string representation of the Event object.
        """
        return f"Event({self.name}, {self.start_time}, {self.end_time}, {self.color}, {self.description})"

    def __str__(self):
        """
        Returns a string representation of the Event object.
        """
        return f"{self.name} ({self.start_time} - {self.end_time})"

    def duration(self):
        """
        Calculates the duration of the event based on the start and end times.

        Returns:
            float: The duration of the event.
        """
        if self.end_time is None:
            return 0
        return self.end_time - self.start_time

    def set_color(self, color):
        """
        Sets the color associated with the event.

        Args:
            color (str): The color to associate with the event.
        """
        self.color = color

    def is_valid(self):
        """
        Checks if the event is valid, based on certain criteria such as the start and end times being in the correct order.

        Returns:
            bool: True if the event is valid, False otherwise.
        """
        if self.end_time is not None and self.start_time > self.end_time:
            return False
        return True

    def to_dict(self):
        """converts the event object to a dictionary for easy serialization and storage in a database"""
        return {
            'name': self.name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'color': self.color,
            'description': self.description,
            'notes': self.notes,
            'location': self.location,
            'participants': self.participants,
        }

    @classmethod
    def from_dict(cls, event_dict):
        """creates an event object from a dictionary"""
        return cls(
            name=event_dict['name'],
            start_time=event_dict['start_time'],
            end_time=event_dict.get('end_time', None),
            color=event_dict.get('color', None),
            description=event_dict.get('description', None),
            notes=event_dict.get('notes', None),
            location=event_dict.get('location', None),
            participants=event_dict.get('participants', None),
        )
