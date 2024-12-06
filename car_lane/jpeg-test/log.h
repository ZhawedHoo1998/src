#ifndef __ALGO_LOG_H
#define __ALGO_LOG_H

#define LOG_COLOR_BLACK   "30"
#define LOG_COLOR_RED     "31"
#define LOG_COLOR_GREEN   "32"
#define LOG_COLOR_BROWN   "33"
#define LOG_COLOR_BLUE    "34"
#define LOG_COLOR_PURPLE  "35"
#define LOG_COLOR_CYAN    "36"
#define LOG_COLOR(COLOR)  "\033[0;" COLOR "m"
#define LOG_BOLD(COLOR)   "\033[1;" COLOR "m"
#define LOG_RESET_COLOR   "\033[0m"
#define LOG_COLOR_E       LOG_COLOR(LOG_COLOR_RED)
#define LOG_COLOR_W       LOG_COLOR(LOG_COLOR_BROWN)
#define LOG_COLOR_I       LOG_COLOR(LOG_COLOR_GREEN)
#define LOG_COLOR_D
#define LOG_COLOR_V

#define LOG_FORMAT(letter, format)  LOG_COLOR_ ## letter #letter " (%" "llu" ") %s: " format LOG_RESET_COLOR //"\n"
#define LOG_FORMAT_ERR(letter, format)  LOG_COLOR_ ## letter #letter " (%" "llu" ") FILE:%s FUNC:%s LINE:%d %s: " format LOG_RESET_COLOR //"\n"
#define LOG_SYSTEM_TIME_FORMAT(letter, format)  LOG_COLOR_ ## letter #letter " (%s) %s: " format LOG_RESET_COLOR //"\n"

typedef enum LOG_LEVEL_ {
  LOG_NONE,   
  LOG_ERR_LEVEL, 
  LOG_WARN_LEVEL, 
  LOG_INFO_LEVEL,
  LOG_DEBUG_LEVEL,
  LOG_VERBOSE_LEVEL
}LOG_LEVEL;


#define LOG_EXPPRT                    __attribute__((visibility("default"))) 

typedef void FUNC_WRITE_LOP_IMP(const char *tag, const char *format, ...);

extern LOG_EXPPRT FUNC_WRITE_LOP_IMP* pfunc_write_log_imp;
#ifdef __cplusplus
extern "C" {
#endif
  LOG_EXPPRT void set_log_level(LOG_LEVEL log_level);
  LOG_EXPPRT LOG_LEVEL get_log_level(void);
  LOG_EXPPRT unsigned long long log_get_timestamp(void);
#ifdef __cplusplus
}
#endif

#define LOG_LEVEL_IMP(level, tag, format, ...)     do {                     \
            switch(level)                                                { \
                case LOG_ERR_LEVEL:pfunc_write_log_imp(tag, LOG_FORMAT_ERR(E, format), log_get_timestamp(),  __FILE__, __FUNCTION__, __LINE__, tag, ##__VA_ARGS__);break; \
                case LOG_WARN_LEVEL:pfunc_write_log_imp(tag, LOG_FORMAT(W, format), log_get_timestamp(), tag, ##__VA_ARGS__);break;\
                case LOG_INFO_LEVEL:pfunc_write_log_imp(tag, LOG_FORMAT(I, format), log_get_timestamp(),  tag, ##__VA_ARGS__);break;\
                case LOG_DEBUG_LEVEL:pfunc_write_log_imp(tag, LOG_FORMAT(D, format), log_get_timestamp(), tag, ##__VA_ARGS__);break;\
                case LOG_VERBOSE_LEVEL:pfunc_write_log_imp(tag, LOG_FORMAT(V, format), log_get_timestamp(), tag, ##__VA_ARGS__);break;\
            } \
} while(0)

#define LOG_LEVEL_LOCAL(level, tag, format, ...) do {               \
        if (get_log_level() >= level ) LOG_LEVEL_IMP(level, tag, format, ##__VA_ARGS__); \
 } while(0)

#define LOGE( tag, format, ... ) LOG_LEVEL_LOCAL(LOG_ERR_LEVEL,   tag, format, ##__VA_ARGS__)
#define LOGW( tag, format, ... ) LOG_LEVEL_LOCAL(LOG_WARN_LEVEL,    tag, format, ##__VA_ARGS__)
#define LOGI( tag, format, ... ) LOG_LEVEL_LOCAL(LOG_INFO_LEVEL,    tag, format, ##__VA_ARGS__)
#define LOGD( tag, format, ... ) LOG_LEVEL_LOCAL(LOG_DEBUG_LEVEL,   tag, format, ##__VA_ARGS__)
#define LOGV( tag, format, ... ) LOG_LEVEL_LOCAL(LOG_VERBOSE_LEVEL, tag, format, ##__VA_ARGS__)

#endif