// 全局变量
let systemInitialized = false;
let currentSearchQuery = '';

// 页面加载完成后执行
$(document).ready(function() {
    // 检查系统状态
    checkSystemStatus();
    
    // 绑定搜索事件
    bindSearchEvents();
    
    // 绑定其他事件
    bindUIEvents();
});

// 检查系统状态
function checkSystemStatus() {
    $.get('/status')
        .done(function(data) {
            updateSystemStatusUI(data);
            if (data.initialized) {
                systemInitialized = true;
                showMoviesSection();
            }
        })
        .fail(function() {
            updateSystemStatusUI({ initialized: false, error: true });
        });
}

// 更新系统状态UI
function updateSystemStatusUI(status) {
    const statusCard = $('#systemStatus');
    const cardBody = statusCard.find('.card-body');
    
    if (status.error) {
        statusCard.removeClass('initialized').addClass('error');
        cardBody.html(`
            <i class="fas fa-exclamation-triangle me-2"></i>
            <span>系统状态检查失败</span>
        `);
    } else if (status.initialized) {
        statusCard.removeClass('error').addClass('initialized');
        cardBody.html(`
            <i class="fas fa-check-circle me-2"></i>
            <span>系统已就绪 - 已加载 ${status.total_movies || 0} 部电影</span>
            ${status.deep_learning_available ? 
                '<small class="d-block mt-1"><i class="fas fa-robot me-1"></i>深度学习功能可用</small>' : 
                '<small class="d-block mt-1"><i class="fas fa-info-circle me-1"></i>深度学习功能不可用</small>'
            }
        `);
    } else {
        statusCard.removeClass('initialized error');
        cardBody.html(`
            <div class="d-flex align-items-center">
                <div class="spinner-border spinner-border-sm me-3" role="status"></div>
                <div>
                    <span>系统未初始化</span>
                    <button class="btn btn-sm btn-primary ms-3" onclick="initializeSystem()">
                        <i class="fas fa-play me-1"></i>初始化系统
                    </button>
                </div>
            </div>
        `);
    }
}

// 初始化系统
function initializeSystem() {
    const statusCard = $('#systemStatus');
    const cardBody = statusCard.find('.card-body');
    
    // 显示加载状态
    cardBody.html(`
        <div class="d-flex align-items-center">
            <div class="spinner-border spinner-border-sm me-3" role="status"></div>
            <span>正在初始化推荐系统，请稍候...</span>
        </div>
    `);
    
    // 发起初始化请求
    $.get('/initialize')
        .done(function(data) {
            if (data.status === 'success') {
                systemInitialized = true;
                showNotification('系统初始化成功！', 'success');
                
                // 重新检查状态
                setTimeout(checkSystemStatus, 1000);
                
                // 显示电影区域
                setTimeout(showMoviesSection, 1500);
            } else {
                showNotification('系统初始化失败', 'error');
                updateSystemStatusUI({ initialized: false, error: true });
            }
        })
        .fail(function() {
            showNotification('系统初始化失败', 'error');
            updateSystemStatusUI({ initialized: false, error: true });
        });
}

// 显示电影区域
function showMoviesSection() {
    if (!systemInitialized) return;
    
    // 加载热门电影
    loadTopMovies();
    
    // 显示电影区域
    $('#moviesSection').fadeIn();
}

// 加载热门电影
function loadTopMovies() {
    const grid = $('#moviesGrid');
    
    // 显示加载状态
    grid.html(`
        <div class="col-12 text-center py-5">
            <div class="spinner-border" role="status">
                <span class="visually-hidden">加载中...</span>
            </div>
            <p class="mt-3 text-muted">正在加载热门电影...</p>
        </div>
    `);
    
    $.get('/top-movies')
        .done(function(movies) {
            displayMovies(movies, grid);
        })
        .fail(function() {
            grid.html(`
                <div class="col-12">
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        加载电影失败，请稍后重试
                    </div>
                </div>
            `);
        });
}

// 显示电影列表
function displayMovies(movies, container) {
    if (!movies || movies.length === 0) {
        container.html(`
            <div class="col-12">
                <div class="alert alert-info">
                    <i class="fas fa-info-circle me-2"></i>
                    暂无电影数据
                </div>
            </div>
        `);
        return;
    }
    
    let html = '';
    movies.forEach(movie => {
        html += createMovieCard(movie);
    });
    
    container.html(html);
    
    // 添加淡入动画
    container.find('.movie-card').addClass('fade-in');
}

// 创建电影卡片HTML
function createMovieCard(movie) {
    const genres = movie.genres || '';
    const genresDisplay = genres.length > 30 ? genres.substring(0, 30) + '...' : genres;
    
    return `
        <div class="col-md-6 col-lg-4 col-xl-3 mb-4">
            <div class="card movie-card h-100">
                <div class="card-body">
                    <h6 class="card-title">
                        <a href="/movie/${encodeURIComponent(movie.title)}" class="text-decoration-none">
                            ${movie.title}
                        </a>
                    </h6>
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <small class="text-muted">${movie.year || 'N/A'}</small>
                        <span class="badge bg-primary">
                            <i class="fas fa-star me-1"></i>${movie.rating || 0}
                        </span>
                    </div>
                    <div class="d-flex justify-content-between align-items-center">
                        <small class="text-muted">
                            ${movie.score ? `推荐分数: ${movie.score}` : ''}
                        </small>
                        <small class="text-muted" title="${genres}">
                            ${genresDisplay}
                        </small>
                    </div>
                </div>
            </div>
        </div>
    `;
}

// 绑定搜索事件
function bindSearchEvents() {
    const searchForm = $('#searchForm');
    const searchInput = $('#searchInput');
    
    // 搜索表单提交
    searchForm.on('submit', function(e) {
        e.preventDefault();
        performSearch();
    });
    
    // 实时搜索（防抖）
    let searchTimeout;
    searchInput.on('input', function() {
        clearTimeout(searchTimeout);
        const query = $(this).val().trim();
        
        if (query.length >= 2) {
            searchTimeout = setTimeout(() => performSearch(query), 500);
        } else if (query.length === 0) {
            hideSearchResults();
        }
    });
}

// 执行搜索
function performSearch(query = null) {
    if (!systemInitialized) {
        showNotification('请先初始化系统', 'warning');
        return;
    }
    
    const searchQuery = query || $('#searchInput').val().trim();
    if (!searchQuery) return;
    
    currentSearchQuery = searchQuery;
    
    // 显示搜索结果区域
    showSearchResults();
    
    const grid = $('#searchGrid');
    
    // 显示加载状态
    grid.html(`
        <div class="col-12 text-center py-3">
            <div class="spinner-border" role="status">
                <span class="visually-hidden">搜索中...</span>
            </div>
            <p class="mt-3 text-muted">正在搜索 "${searchQuery}"...</p>
        </div>
    `);
    
    $.get('/search', { q: searchQuery })
        .done(function(results) {
            displaySearchResults(results, searchQuery);
        })
        .fail(function() {
            grid.html(`
                <div class="col-12">
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        搜索失败，请稍后重试
                    </div>
                </div>
            `);
        });
}

// 显示搜索结果
function displaySearchResults(results, query) {
    const grid = $('#searchGrid');
    const section = $('#searchResults');
    
    if (!results || results.length === 0) {
        grid.html(`
            <div class="col-12">
                <div class="alert alert-info">
                    <i class="fas fa-search me-2"></i>
                    没有找到与 "${query}" 相关的电影
                </div>
            </div>
        `);
        return;
    }
    
    let html = '';
    results.forEach(movie => {
        html += createSearchResultCard(movie);
    });
    
    grid.html(html);
    
    // 添加动画
    grid.find('.movie-card').addClass('fade-in');
    
    // 滚动到搜索结果
    section[0].scrollIntoView({ behavior: 'smooth' });
}

// 创建搜索结果卡片
function createSearchResultCard(movie) {
    const overview = movie.overview || '';
    const overviewDisplay = overview.length > 100 ? overview.substring(0, 100) + '...' : overview;
    
    return `
        <div class="col-md-6 col-lg-4 mb-4">
            <div class="card movie-card h-100">
                <div class="card-body">
                    <h6 class="card-title">
                        <a href="/movie/${encodeURIComponent(movie.title)}" class="text-decoration-none">
                            ${movie.title}
                        </a>
                    </h6>
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <small class="text-muted">${movie.year || 'N/A'}</small>
                        <span class="badge bg-primary">
                            <i class="fas fa-star me-1"></i>${movie.rating || 0}
                        </span>
                    </div>
                    ${overviewDisplay ? `<p class="card-text small text-muted">${overviewDisplay}</p>` : ''}
                    <small class="text-muted">${movie.genres || ''}</small>
                </div>
            </div>
        </div>
    `;
}

// 显示搜索结果区域
function showSearchResults() {
    $('#searchResults').fadeIn();
    $('#moviesSection').hide();
}

// 隐藏搜索结果区域
function hideSearchResults() {
    $('#searchResults').hide();
    $('#moviesSection').fadeIn();
}

// 绑定UI事件
function bindUIEvents() {
    // 开始探索按钮
    $('#startBtn').on('click', function() {
        if (systemInitialized) {
            $('html, body').animate({
                scrollTop: $('#moviesSection').offset().top - 100
            }, 800);
        } else {
            initializeSystem();
        }
    });
    
    // 了解更多按钮
    $('#aboutBtn').on('click', function() {
        $('html, body').animate({
            scrollTop: $('.features-section').offset().top - 100
        }, 800);
    });
    
    // 加载更多按钮
    $('#loadMoreBtn').on('click', function() {
        // 这里可以实现分页加载更多电影
        showNotification('功能开发中...', 'info');
    });
}

// 显示通知
function showNotification(message, type = 'info') {
    const alertClass = `alert-${type === 'error' ? 'danger' : type}`;
    const iconClass = type === 'success' ? 'check-circle' : 
                     type === 'error' ? 'exclamation-triangle' : 
                     type === 'warning' ? 'exclamation-circle' : 'info-circle';
    
    const notification = $(`
        <div class="alert ${alertClass} alert-dismissible fade show position-fixed" 
             style="top: 20px; right: 20px; z-index: 9999; min-width: 300px;">
            <i class="fas fa-${iconClass} me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `);
    
    $('body').append(notification);
    
    // 自动消失
    setTimeout(() => {
        notification.alert('close');
    }, 5000);
}

// 工具函数：格式化数字
function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

// 工具函数：截断文本
function truncateText(text, maxLength) {
    if (!text) return '';
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
}
