package ops

import "os"

func CheckDir(d string) bool {
	_, err := os.Stat(d)
	return os.IsNotExist(err)
}

func CleanDir(ds ...string) {
	for _, d := range ds {
		err := os.RemoveAll(d)
		CheckE(err)
		err = os.MkdirAll(d, 0755)
		CheckE(err)
	}
}
