cmake_minimum_required(VERSION 3.16)
project(lasso_cd C)

set(CMAKE_C_STANDARD 11)

# --- Core files ---
set(CORE_SRC
        core/cd_engine.c
        core/state.h
)

# --- Rule modules ---
set(RULES_SRC
        rules/rule_cyclic.c
        rules/rule_shuffle.c
        rules/rule_random.c
        rules/rule_block_shuffle.c
        rules/rule_gs_s.c
        rules/rule_gs_r.c          # <‑‑ NEW
)


# --- Update schemes ---
set(UPDATES_SRC
        updates/upd_prox_linear.c
        updates/upd_prox_point.c
        updates/upd_bcm.c
        updates/upd_prox_linear_ext.c
        updates/upd_prox_linear_sgd.c
        updates/upd_prox_linear_svrg.c
        updates/upd_prox_linear_enet.c       # NEW
)





# --- Utils and I/O ---
set(UTILS_SRC
        utils.c
        io.c
)

# --- Build shared library ---
add_library(lasso_cd SHARED
        ${CORE_SRC}
        ${RULES_SRC}
        ${UPDATES_SRC}
        ${UTILS_SRC}
        cd_api.c
)

target_include_directories(lasso_cd PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/core
        ${CMAKE_CURRENT_SOURCE_DIR}/rules
        ${CMAKE_CURRENT_SOURCE_DIR}/updates
        ${CMAKE_CURRENT_SOURCE_DIR}  # for utils/io headers
)

target_link_libraries(lasso_cd m)  # link libm for math
