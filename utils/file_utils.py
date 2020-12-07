class FileWrite(object):

    def __init__(self, file_name,  write_to_file=True, clean_file=True):
        super().__init__()
        self.write = write_to_file
        self.file_name = file_name
        if clean_file:
            self.clean_file()

    def clean_file(self):
        if self.write:
            file_h = open(self.file_name, "a")
            file_h.truncate(0)
            file_h.close()

    def write_to_file(self, input_str):
        file_h = open(self.file_name, "a")
        file_h.write(input_str)
        file_h.close()

