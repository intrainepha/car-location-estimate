package ops

import (
	"strconv"
)

func Str2int(str string) int {
	/*Convert string to int

	Args:
		str(string): string data

	Returns:
		floatNum(float64): int data
	*/

	intNum, err := strconv.Atoi(str)
	CheckE(err)

	return intNum
}

func Str2f64(str string) float64 {
	/*Convert string to float64

	Args:
		str(string): string data

	Returns:
		intNum(float64): float64 data
	*/

	floatNum, err := strconv.ParseFloat(str, 64)
	CheckE(err)

	return floatNum
}

func F642Str(num float64) string {
	/*Convert float64 to string

	Args:
		num(float64): float64 data

	Returns:
		(string): string data
	*/

	return strconv.FormatFloat(num, 'g', -1, 64)
}
