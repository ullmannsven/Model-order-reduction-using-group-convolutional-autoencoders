class ProgressBar:

    def __init__(self, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        self.print_end = print_end
        self.iteration = 0

        self.update()

    def update(self):
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (self.iteration / float(self.total)))
        filled_length = int(self.length * self.iteration // self.total)
        bar = self.fill * filled_length + '-' * (self.length - filled_length)
        print('\r%s |%s| %s%% %s' % (self.prefix, bar, percent, self.suffix), end=self.print_end)
        if self.iteration == self.total:
            print()
        self.iteration = self.iteration + 1


class ProgressTraining:

    def __init__(self, total, prefix='', suffix='', decimals=1, print_end='\r'):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.iteration = 0
        self.print_end = print_end

    def update(self, training_loss=None, validation_loss=None):
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (self.iteration / float(self.total)))
        if training_loss is None:
            print('\r{} {}% {}'.format(self.prefix, percent, self.suffix), end=self.print_end)
        elif validation_loss is None:
            print('\r{} {}% {}; TL: {:.4E}'.format(self.prefix, percent, self.suffix, training_loss), end=self.print_end)
        else:
            print('\r{} {}% {}; TL: {:.4E}; VL: {:.4E}'.format(self.prefix, percent, self.suffix, training_loss, validation_loss), end=self.print_end)

        if self.iteration == self.total:
            print()
        self.iteration = self.iteration + 1
