#define C_BLACK "\033[0;30m"
#define C_RED "\033[0;31m"
#define C_GREEN "\033[0;32m"
#define C_YELLOW "\033[0;33m"
#define C_BLUE "\033[0;34m"
#define C_PURPLE "\033[0;35m"
#define C_CYAN "\033[0;36m"
#define C_WHITE "\033[0;37m"
#define C_RESET "\033[0m"

#define C_BGRED "\033[0;41m"

#define BLUE_TEXT(TXT) C_BLUE TXT C_RESET
#define GREEN_TEXT(TXT) C_GREEN TXT C_RESET
#define RED_TEXT(TXT) C_RED TXT C_RESET
#define COLOR_TEXT(TXT,CLR) CLR TXT C_RESET
