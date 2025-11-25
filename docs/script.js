let currentApplication = 0;
const applicationSlides = document.querySelectorAll('.application-slide');
const applicationIndicators = document.querySelectorAll('.applications .indicator');

function changeApplication(direction) {
    applicationSlides[currentApplication].classList.remove('active');
    applicationIndicators[currentApplication].classList.remove('active');
    
    currentApplication += direction;
    
    if (currentApplication >= applicationSlides.length) {
        currentApplication = 0;
    } else if (currentApplication < 0) {
        currentApplication = applicationSlides.length - 1;
    }
    
    applicationSlides[currentApplication].classList.add('active');
    applicationIndicators[currentApplication].classList.add('active');
}

function setApplication(index) {
    applicationSlides[currentApplication].classList.remove('active');
    applicationIndicators[currentApplication].classList.remove('active');
    
    currentApplication = index;
    
    applicationSlides[currentApplication].classList.add('active');
    applicationIndicators[currentApplication].classList.add('active');
}

let currentAnalysis = 0;
const analysisSlides = document.querySelectorAll('.analysis-slide');
const analysisIndicators = document.querySelectorAll('.analysis .indicator');

function changeAnalysis(direction) {
    analysisSlides[currentAnalysis].classList.remove('active');
    analysisIndicators[currentAnalysis].classList.remove('active');
    
    currentAnalysis += direction;
    
    if (currentAnalysis >= analysisSlides.length) {
        currentAnalysis = 0;
    } else if (currentAnalysis < 0) {
        currentAnalysis = analysisSlides.length - 1;
    }
    
    analysisSlides[currentAnalysis].classList.add('active');
    analysisIndicators[currentAnalysis].classList.add('active');
}

function setAnalysis(index) {
    analysisSlides[currentAnalysis].classList.remove('active');
    analysisIndicators[currentAnalysis].classList.remove('active');
    
    currentAnalysis = index;
    
    analysisSlides[currentAnalysis].classList.add('active');
    analysisIndicators[currentAnalysis].classList.add('active');
}

// Smooth scroll navigation with navbar offset
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const targetId = this.getAttribute('href');
        const target = document.querySelector(targetId);
        if (target) {
            const navbarHeight = 70;
            const extraPadding = 20;
            const targetPosition = target.getBoundingClientRect().top + window.pageYOffset - navbarHeight - extraPadding;
            
            window.scrollTo({
                top: targetPosition,
                behavior: 'smooth'
            });
        }
    });
});

// Navbar scroll effect
window.addEventListener('scroll', function() {
    const navbar = document.querySelector('.navbar');
    if (window.scrollY > 100) {
        navbar.style.background = 'rgba(255, 255, 255, 0.98)';
        navbar.style.boxShadow = '0 2px 20px rgba(0, 0, 0, 0.1)';
    } else {
        navbar.style.background = 'rgba(255, 255, 255, 0.95)';
        navbar.style.boxShadow = 'none';
    }
});

// Enhanced video lazy loading and streaming functionality
function initVideoStreaming() {
    // Main video streaming with progress
    const mainVideo = document.getElementById('mainVideo');
    const mainVideoLoader = document.getElementById('mainVideoLoader');
    const mainVideoProgress = document.getElementById('mainVideoProgress');
    
    if (mainVideo && mainVideoLoader) {
        // Create Intersection Observer for lazy loading
        const videoObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    loadVideoWithProgress(mainVideo, mainVideoLoader, mainVideoProgress);
                    videoObserver.unobserve(mainVideo);
                }
            });
        }, {
            rootMargin: '100px' // Start loading 100px before entering viewport
        });
        
        videoObserver.observe(mainVideo);
    }
    
    // Lazy load other videos
    lazyLoadVideos();
}

function loadVideoWithProgress(video, loader, progressBar) {
    const videoSrc = video.dataset.src;
    if (!videoSrc) return;
    
    // Show loader
    loader.classList.remove('hidden');
    
    // Create XMLHttpRequest to track loading progress
    const xhr = new XMLHttpRequest();
    xhr.open('GET', videoSrc, true);
    xhr.responseType = 'blob';
    
    // Track download progress
    xhr.onprogress = function(e) {
        if (e.lengthComputable) {
            const percentComplete = (e.loaded / e.total) * 100;
            if (progressBar) {
                progressBar.style.width = percentComplete + '%';
            }
        }
    };
    
    // On load complete
    xhr.onload = function() {
        if (xhr.status === 200) {
            const blob = xhr.response;
            const blobUrl = URL.createObjectURL(blob);
            
            // Set video source
            const source = video.querySelector('source');
            if (source) {
                source.src = blobUrl;
            }
            video.src = blobUrl;
            
            // Wait for video to be ready to play
            video.addEventListener('loadeddata', function() {
                // Hide loader
                setTimeout(() => {
                    loader.classList.add('hidden');
                    video.classList.add('loaded');
                }, 300);
                
                // Attempt autoplay
                const playPromise = video.play();
                if (playPromise !== undefined) {
                    playPromise.catch(error => {
                        // Autoplay was prevented, user needs to interact
                        console.log('Autoplay prevented:', error);
                    });
                }
            });
            
            // Load video
            video.load();
        }
    };
    
    // Error handling
    xhr.onerror = function() {
        console.error('Video loading failed');
        loader.querySelector('.loader-text').textContent = 'Loading failed, please refresh';
        if (progressBar) {
            progressBar.style.background = '#ef4444';
            progressBar.style.width = '100%';
        }
    };
    
    // Send request
    xhr.send();
}

// Standard lazy loading for other videos
function lazyLoadVideos() {
    const videos = document.querySelectorAll('video.lazy-video:not(#mainVideo)');
    
    const videoObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const video = entry.target;
                
                // Check if already loaded or loading
                if (video.dataset.isLoading === 'true' || video.dataset.isLoaded === 'true') {
                    console.log('Video already loading or loaded, skipping:', video.dataset.src);
                    return;
                }
                
                // Mark as loading
                video.dataset.isLoading = 'true';
                
                const source = video.querySelector('source[data-src]');
                
                // Find the loader element
                const loader = video.parentElement.querySelector('.video-loader');
                
                console.log('Starting to load video:', video.dataset.src);
                
                // Show loader
                if (loader) {
                    loader.classList.remove('hidden');
                }
                
                // Set up load event handler
                const onLoad = function() {
                    console.log('Video loaded successfully:', video.dataset.src);
                    video.dataset.isLoading = 'false';
                    video.dataset.isLoaded = 'true';
                    
                    setTimeout(() => {
                        if (loader) {
                            loader.classList.add('hidden');
                        }
                        video.classList.add('loaded');
                    }, 300);
                };
                
                const onError = function() {
                    console.error('Video failed to load:', video.dataset.src);
                    video.dataset.isLoading = 'false';
                    if (loader) {
                        loader.querySelector('.loader-text').textContent = 'Failed to load video';
                    }
                };
                
                video.addEventListener('loadeddata', onLoad, { once: true });
                video.addEventListener('error', onError, { once: true });
                
                // Set video sources
                if (source && source.dataset.src) {
                    source.src = source.dataset.src;
                    source.removeAttribute('data-src');
                }
                
                if (video.dataset.src) {
                    const videoSrc = video.dataset.src;
                    video.src = videoSrc;
                    video.removeAttribute('data-src');
                }
                
                video.load();
                
                // Stop observing this video
                videoObserver.unobserve(video);
            }
        });
    }, {
        rootMargin: '100px'
    });
    
    videos.forEach(video => {
        videoObserver.observe(video);
    });
}

// Initialize video streaming on page load
document.addEventListener('DOMContentLoaded', initVideoStreaming);

// Switch iframe content for interactive examples
function switchIframe(element) {
    console.log('switchIframe called');
    
    // Get the iframe src from data attribute
    const newSrc = element.getAttribute('data-src');
    const mainIframe = document.getElementById('mainIframe');
    const iframeLoader = document.getElementById('iframeLoader');
    
    console.log('New src:', newSrc);
    console.log('Main iframe:', mainIframe);
    console.log('Loader:', iframeLoader);
    
    if (!mainIframe) {
        console.error('Main iframe not found!');
        return;
    }
    
    if (!newSrc) {
        console.error('No data-src attribute found!');
        return;
    }
    
    // Don't reload if it's the same src
    if (mainIframe.src === newSrc) {
        console.log('Same src, skipping reload');
        return;
    }
    
    console.log('Starting iframe switch...');
    
    // Show loader and hide iframe
    if (iframeLoader) {
        iframeLoader.classList.remove('hidden');
        iframeLoader.querySelector('.loader-text').textContent = 'Loading interactive demo...';
    }
    mainIframe.classList.remove('loaded');
    
    // Update active state on thumbnails
    const thumbnails = document.querySelectorAll('.thumbnail-item');
    thumbnails.forEach(thumb => {
        thumb.classList.remove('active', 'loading');
    });
    element.classList.add('active', 'loading');
    
    // Update iframe src
    console.log('Setting iframe src to:', newSrc);
    mainIframe.src = newSrc;
}

// Initialize iframe loading handlers
document.addEventListener('DOMContentLoaded', function() {
    const mainIframe = document.getElementById('mainIframe');
    const iframeLoader = document.getElementById('iframeLoader');
    
    if (mainIframe && iframeLoader) {
        console.log('Setting up iframe event listeners');
        
        // Handle iframe load event
        mainIframe.addEventListener('load', function() {
            console.log('Iframe loaded successfully');
            setTimeout(() => {
                iframeLoader.classList.add('hidden');
                mainIframe.classList.add('loaded');
                
                // Remove loading state from all thumbnails
                const thumbnails = document.querySelectorAll('.thumbnail-item');
                thumbnails.forEach(thumb => {
                    thumb.classList.remove('loading');
                });
            }, 500);
        });
        
        // Handle iframe error event
        mainIframe.addEventListener('error', function() {
            console.error('Iframe failed to load');
            if (iframeLoader) {
                iframeLoader.querySelector('.loader-text').textContent = 'Failed to load. Please try again.';
            }
            
            // Remove loading state from all thumbnails
            const thumbnails = document.querySelectorAll('.thumbnail-item');
            thumbnails.forEach(thumb => {
                thumb.classList.remove('loading');
            });
        });
        
        // Initial load - show iframe after it loads
        console.log('Waiting for initial iframe load...');
    } else {
        console.error('Iframe or loader not found!', { mainIframe, iframeLoader });
    }
});

document.addEventListener('DOMContentLoaded', function() {
    const elements = document.querySelectorAll('.overview-card, .demo-item, .model-card');
    elements.forEach((element, index) => {
        element.style.opacity = '0';
        element.style.transform = 'translateY(30px)';
        
        setTimeout(() => {
            element.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
            element.style.opacity = '1';
            element.style.transform = 'translateY(0)';
        }, index * 100);
    });
});

document.addEventListener('keydown', function(e) {
    if (e.key === 'ArrowLeft') {
        changeApplication(-1);
        changeAnalysis(-1);
    } else if (e.key === 'ArrowRight') {
        changeApplication(1);
        changeAnalysis(1);
    }
});

let touchStartX = 0;
let touchEndX = 0;

document.addEventListener('touchstart', function(e) {
    touchStartX = e.changedTouches[0].screenX;
});

document.addEventListener('touchend', function(e) {
    touchEndX = e.changedTouches[0].screenX;
    handleSwipe();
});

function handleSwipe() {
    const swipeThreshold = 50;
    const diff = touchStartX - touchEndX;
    
    if (Math.abs(diff) > swipeThreshold) {
        if (diff > 0) {
            changeApplication(1);
            changeAnalysis(1);
        } else {
            changeApplication(-1);
            changeAnalysis(-1);
        }
    }
}

// Copy behavior initialization
function initCitationCopy() {
    const btn = document.querySelector('.copy-code-btn');
    const code = document.querySelector('.citation-box pre code');
    if (!btn || !code) return;

    btn.addEventListener('click', () => {
        const text = code.textContent.trim();
        navigator.clipboard.writeText(text).then(() => {
            btn.classList.add('copied');
            const originalHTML = btn.innerHTML;
            btn.innerHTML = '<i class="fas fa-check"></i><span>Copied</span>';
            setTimeout(() => {
                btn.classList.remove('copied');
                btn.innerHTML = originalHTML;
            }, 2000);
        }).catch(() => {
            btn.innerHTML = '<i class="fas fa-exclamation-triangle"></i><span>Error</span>';
            setTimeout(() => {
                btn.innerHTML = '<i class="fas fa-copy"></i><span>Copy</span>';
            }, 2000);
        });
    });
}

document.addEventListener('DOMContentLoaded', initCitationCopy);
