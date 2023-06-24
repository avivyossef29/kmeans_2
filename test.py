import subprocess
import unittest

FILE11 = "input_1_db_1.txt"
FILE21 = "input_1_db_2.txt"

FILE12 = "input_2_db_1.txt"
FILE22 = "input_2_db_2.txt"

FILE13 = "input_3_db_1.txt"
FILE23 = "input_3_db_2.txt"

OUT1 = "output_1.txt"
OUT2 = "output_2.txt"
OUT3 = "output_3.txt"

DEFAULT_ITER = 300
DEFUALT_EPS = 0.01
STANDARD_ERROR = "An Error Has Occurred"
ITER_ERROR = "Invalid maximum iteration!"
CLUSTER_ERROR = "Invalid number of clusters!"


def runPy(args, file1=FILE11, file2=FILE21):
    cmd = 'python3 kmeans_pp.py'
    for arg in args:
        cmd += " " + str(arg)

    cmd += " " + file1 + " " + file2
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    return p.communicate()


def fetchError(str, errorCode):
    if (errorCode == 0):
        return (str.count(CLUSTER_ERROR) == 1)
    elif (errorCode == 1):
        return (str.count(ITER_ERROR) == 1)
    elif (errorCode == 2):
        return (str.count(STANDARD_ERROR) == 1)
    return None


def formatOut(str):
    if (str[-1] == "\n"):
        str = str[0:-1]
    return str


class TestKmeansMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._testMatrix = []

        f = open("tests.txt", "r")
        data = f.readlines()
        for line in data:
            line = formatOut(line)
            lineTup = line.split(" ")

            lineTup[-1] = str(lineTup[-1][1:-1]).split(",")
            cls._testMatrix += [lineTup]

        f.close()

    def test_File(self):
        for i, testLine in enumerate(self._testMatrix):
            with self.subTest(i=i, msg=(str(testLine) + " Error")):
                com = runPy(testLine[1:-1])
                out, err = com
                outF = formatOut(out.decode("utf-8"))
                for errorCode in testLine[-1]:
                    self.assertTrue(fetchError(outF, int(errorCode)))

    def test_properInput(self):
        out1, err1 = runPy([3, 333, 0], FILE11, FILE21)
        out2, err2 = runPy([7, 750, 0], FILE12, FILE22)
        out3, err3 = runPy([15, 750, 0], FILE13, FILE23)
        out1 = out1.decode("utf-8")
        out2 = out2.decode("utf-8")
        out3 = out3.decode("utf-8")

        fo1 = open(OUT1, "r")
        fo2 = open(OUT2, "r")
        fo3 = open(OUT3, "r")
        data1 = fo1.read()
        data2 = fo2.read()
        data3 = fo3.read()
        fo1.close()
        fo2.close()
        fo3.close()
        self.maxDiff = None
        self.assertEqual(data1, out1)
        self.assertEqual(data2, out2)
        self.assertEqual(data3, out3)

        out4, err4 = runPy([6, 117, 0.001], FILE11, FILE21)
        allOuts = out4.decode("utf-8")
        out4, err4 = runPy([30, 555, 0], FILE12, FILE22)
        allOuts += out4.decode("utf-8")
        out4, err4 = runPy([3, 666, 0.117], FILE13, FILE23)
        allOuts += out4.decode("utf-8")
        out4, err4 = runPy([10, 0], FILE11, FILE21)
        allOuts += out4.decode("utf-8")
        out4, err4 = runPy([3, 42, 0.5], FILE13, FILE23)
        allOuts += out4.decode("utf-8")
        out4, err4 = runPy([6, 999, 0.0001], FILE12, FILE22)
        allOuts += out4.decode("utf-8")
        out4, err4 = runPy([9, 10, 0], FILE11, FILE21)
        allOuts += out4.decode("utf-8")
        fo4 = open('newOut.txt', "r")
        data4 = fo4.read()
        fo4.close()
        self.assertEqual(data4, allOuts)


if __name__ == '__main__':
    unittest.main()
