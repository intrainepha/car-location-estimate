package ops

import "os"

func CheckDir(d string) bool {
	_, err := os.Stat(d)
	return os.IsNotExist(err)
}

func CleanDir(ds ...string) {
	for _, d := range ds {
		os.RemoveAll(d)
		os.MkdirAll(d, 0755)
	}
}
