// 首页专用JavaScript

$(document).ready(function() {
    // 首页特有的初始化逻辑
    initializeHomePage();
    
    // 动态效果
    addDynamicEffects();
});

// 初始化首页
function initializeHomePage() {
    // 检查系统状态并相应地显示内容
    setTimeout(checkAndDisplayContent, 1000);
    
    // 添加滚动效果
    addScrollEffects();
}

// 检查并显示内容
function checkAndDisplayContent() {
    if (systemInitialized) {
        // 如果系统已初始化，直接显示电影
        showMoviesSection();
    } else {
        // 监听系统初始化状态
        const checkInterval = setInterval(() => {
            if (systemInitialized) {
                showMoviesSection();
                clearInterval(checkInterval);
            }
        }, 1000);
    }
}

// 添加动态效果
function addDynamicEffects() {
    // Hero区域动画
    animateHeroSection();
    
    // 功能卡片悬停效果
    enhanceFeatureCards();
    
    // 数字动画效果
    animateNumbers();
}

// Hero区域动画
function animateHeroSection() {
    const heroIcon = $('.hero-icon');
    
    // 鼠标移动效果
    $('.hero-section').on('mousemove', function(e) {
        const x = (e.pageX - $(this).offset().left) / $(this).width();
        const y = (e.pageY - $(this).offset().top) / $(this).height();
        
        heroIcon.css({
            'transform': `translate(${x * 20 - 10}px, ${y * 20 - 10}px) rotateY(${x * 20 - 10}deg)`
        });
    });
    
    // 鼠标离开时重置
    $('.hero-section').on('mouseleave', function() {
        heroIcon.css({
            'transform': 'translate(0, 0) rotateY(0deg)'
        });
    });
}

// 增强功能卡片
function enhanceFeatureCards() {
    $('.feature-card').each(function(index) {
        const card = $(this);
        
        // 延迟显示动画
        setTimeout(() => {
            card.addClass('fade-in');
        }, index * 200);
        
        // 3D倾斜效果
        card.on('mouseenter', function(e) {
            const rect = this.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            const rotateX = (y - centerY) / 10;
            const rotateY = (centerX - x) / 10;
            
            $(this).css({
                'transform': `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateZ(10px)`
            });
        });
        
        card.on('mouseleave', function() {
            $(this).css({
                'transform': 'perspective(1000px) rotateX(0) rotateY(0) translateZ(0)'
            });
        });
    });
}

// 数字动画效果
function animateNumbers() {
    // 当系统状态更新时，动画显示电影数量
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList') {
                const statusText = $('#systemStatus .card-body').text();
                const match = statusText.match(/(\d+)\s*部电影/);
                if (match) {
                    const totalMovies = parseInt(match[1]);
                    animateCounter(totalMovies);
                }
            }
        });
    });
    
    observer.observe(document.getElementById('systemStatus'), {
        childList: true,
        subtree: true
    });
}

// 计数器动画
function animateCounter(target) {
    const duration = 2000; // 2秒
    const start = 0;
    const startTime = performance.now();
    
    function updateCounter(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // 使用缓动函数
        const easeOut = 1 - Math.pow(1 - progress, 3);
        const current = Math.floor(start + (target - start) * easeOut);
        
        // 更新显示（如果有计数器元素的话）
        $('.movie-counter').text(current.toLocaleString());
        
        if (progress < 1) {
            requestAnimationFrame(updateCounter);
        }
    }
    
    requestAnimationFrame(updateCounter);
}

// 添加滚动效果
function addScrollEffects() {
    // 滚动时的视差效果
    $(window).on('scroll', function() {
        const scrollTop = $(this).scrollTop();
        const heroSection = $('.hero-section');
        
        // 视差背景效果
        heroSection.css({
            'transform': `translateY(${scrollTop * 0.5}px)`
        });
        
        // 导航栏背景透明度
        const navbar = $('.navbar');
        if (scrollTop > 100) {
            navbar.addClass('scrolled');
        } else {
            navbar.removeClass('scrolled');
        }
    });
    
    // 元素进入视图时的动画
    const observeElements = () => {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-in');
                }
            });
        }, {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        });
        
        // 观察需要动画的元素
        document.querySelectorAll('.feature-card, .movie-card').forEach(el => {
            observer.observe(el);
        });
    };
    
    // 页面加载完成后开始观察
    setTimeout(observeElements, 1000);
}

// 添加粒子效果（可选）
function addParticleEffect() {
    const canvas = $('<canvas id="particles"></canvas>');
    canvas.css({
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        'pointer-events': 'none',
        'z-index': 0
    });
    
    $('.hero-section').prepend(canvas);
    
    // 简单的粒子系统
    const ctx = canvas[0].getContext('2d');
    const particles = [];
    
    // 初始化粒子
    for (let i = 0; i < 50; i++) {
        particles.push({
            x: Math.random() * canvas.width(),
            y: Math.random() * canvas.height(),
            vx: (Math.random() - 0.5) * 0.5,
            vy: (Math.random() - 0.5) * 0.5,
            size: Math.random() * 2 + 1,
            opacity: Math.random() * 0.5 + 0.2
        });
    }
    
    // 动画循环
    function animateParticles() {
        ctx.clearRect(0, 0, canvas.width(), canvas.height());
        
        particles.forEach(particle => {
            // 更新位置
            particle.x += particle.vx;
            particle.y += particle.vy;
            
            // 边界检查
            if (particle.x < 0 || particle.x > canvas.width()) particle.vx *= -1;
            if (particle.y < 0 || particle.y > canvas.height()) particle.vy *= -1;
            
            // 绘制粒子
            ctx.globalAlpha = particle.opacity;
            ctx.fillStyle = 'white';
            ctx.beginPath();
            ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
            ctx.fill();
        });
        
        requestAnimationFrame(animateParticles);
    }
    
    // 启动动画
    animateParticles();
}

// 键盘快捷键
$(document).on('keydown', function(e) {
    // 按 '/' 键聚焦搜索框
    if (e.key === '/' && !$(e.target).is('input, textarea')) {
        e.preventDefault();
        $('#searchInput').focus();
    }
    
    // 按 Escape 键清除搜索
    if (e.key === 'Escape') {
        $('#searchInput').val('').blur();
        hideSearchResults();
    }
});

// 添加加载进度指示器
function showLoadingProgress() {
    const progressBar = $(`
        <div class="loading-progress position-fixed" style="top: 0; left: 0; right: 0; z-index: 9999;">
            <div class="progress" style="height: 3px; border-radius: 0;">
                <div class="progress-bar bg-primary" style="width: 0%; transition: width 0.3s ease;"></div>
            </div>
        </div>
    `);
    
    $('body').prepend(progressBar);
    
    // 模拟进度
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) progress = 90;
        
        progressBar.find('.progress-bar').css('width', progress + '%');
        
        if (systemInitialized) {
            progressBar.find('.progress-bar').css('width', '100%');
            setTimeout(() => {
                progressBar.fadeOut(() => progressBar.remove());
            }, 500);
            clearInterval(interval);
        }
    }, 200);
}
