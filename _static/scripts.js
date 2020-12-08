var $window = $(window),
    $head = $('head'),
    $body = $('body'),
    $sidebar = $('.sidebar'),
    $sidebarToggle = $('.btn__sidebar');

function setContrast() {
    var setContrast = localStorage.setContrast;
    if (setContrast == 1) {
        $body.addClass('dark-theme');
        $('.btn__contrast').addClass('btn-active');
    }
}

setContrast();

$('.btn__contrast').on('click', function (event) {
    // Prevent default.
    event.preventDefault();
    event.stopPropagation();

    if ($(this).hasClass('btn-active')) {
        $(this).removeClass('btn-active');
        localStorage.setContrast = 0;
        $body.removeClass('dark-theme');
    } else {
        $(this).addClass('btn-active');
        localStorage.setContrast = 1;
        $body.addClass('dark-theme');
    }
});

// Tooltips

tippy('[data-tippy-content]', {
    touch: false,
});

// Icons

feather.replace();

// Sidebar toggles

function openSidebar() {
    $sidebarToggle.addClass('btn-active');
    $sidebar.removeClass('inactive');
    $(".toolbar svg.feather.feather-menu").replaceWith(feather.icons.x.toSvg());
}
function closeSidebar() {
    $sidebarToggle.removeClass('btn-active');
    $sidebar.addClass('inactive');
    $(".toolbar svg.feather.feather-x").replaceWith(feather.icons.menu.toSvg());
}

$sidebarToggle.on('click', function (event) {
    event.preventDefault();
    event.stopPropagation();
    if ($sidebar.hasClass('inactive')) {
        openSidebar();
    } else {
        closeSidebar();
    }
    if (window.innerWidth <= 1340) {
        $(document.body).on('click', function (e) {
            if (!$(event.target).is('.sidebar *')) {
                closeSidebar();
                $body.off('click');
            }
        });
    }
});

$('.btn__top').on('click', function (event) {
    event.preventDefault();
    event.stopPropagation();
    $('html, body').animate({ scrollTop: 0 }, 'slow');
});

$('.btn__fullscreen').on('click', function () {
    event.preventDefault();
    event.stopPropagation();
    $(this).toggleClass('btn-active');

    if (document.fullscreenElement || document.webkitFullscreenElement || document.mozFullScreenElement || document.msFullscreenElement) {
        //in fullscreen, so exit it
        if (document.exitFullscreen) {
            document.exitFullscreen();
        } else if (document.msExitFullscreen) {
            document.msExitFullscreen();
        } else if (document.mozCancelFullScreen) {
            document.mozCancelFullScreen();
        } else if (document.webkitExitFullscreen) {
            document.webkitExitFullscreen();
        }
    } else {
        //not fullscreen, so enter it
        if (document.documentElement.requestFullscreen) {
            document.documentElement.requestFullscreen();
        } else if (document.documentElement.webkitRequestFullscreen) {
            document.documentElement.webkitRequestFullscreen();
        } else if (document.documentElement.mozRequestFullScreen) {
            document.documentElement.mozRequestFullScreen();
        } else if (document.documentElement.msRequestFullscreen) {
            document.documentElement.msRequestFullscreen();
        }
    }
});

function setFontSize() {
    // Get font size from local storage
    var toolbarFont = localStorage.toolbarFont;
    if (toolbarFont == 1) {
        $('html').addClass('font-plus');
    } else if (toolbarFont == -1) {
        $('html').addClass('font-minus');
    } else {
        $('html').removeClass('font-plus');
        $('html').removeClass('font-minus');
        localStorage.toolbarFont = 0;
    }
}

setFontSize();

$('.btn__plus').on('click', function (event) {
    event.preventDefault();
    event.stopPropagation();
    var toolbarFont = parseInt(localStorage.getItem('toolbarFont')) + 1;
    if (toolbarFont > 0) {
        toolbarFont = 1;
    }
    localStorage.toolbarFont = toolbarFont;
    setFontSize();
});

$('.btn__minus').on('click', function (event) {
    event.preventDefault();
    event.stopPropagation();
    var toolbarFont = parseInt(localStorage.getItem('toolbarFont')) - 1;
    if (toolbarFont < 0) {
        toolbarFont = -1;
    }
    localStorage.toolbarFont = toolbarFont;
    setFontSize();
});

// Declare MathJax Macros for the Appropriate Macros
MathJax.Hub.Config({
    TeX: {
        Macros: {
            Var: '\\mathop{\\mathrm{Var}}',
            trace: '\\mathop{\\mathrm{trace}}',
            argmax: '\\mathop{\\mathrm{arg\\,max}}',
            argmin: '\\mathop{\\mathrm{arg\\,min}}',
            proj: '\\mathop{\\mathrm{proj}}',
            col: '\\mathop{\\mathrm{col}}',
            Span: '\\mathop{\\mathrm{span}}',
            epsilon: '\\varepsilon',
            EE: '\\mathbb{E}',
            PP: '\\mathbb{P}',
            RR: '\\mathbb{R}',
            NN: '\\mathbb{N}',
            ZZ: '\\mathbb{Z}',
            aA: '\\mathcal{A}',
            bB: '\\mathcal{B}',
            cC: '\\mathcal{C}',
            dD: '\\mathcal{D}',
            eE: '\\mathcal{E}',
            fF: '\\mathcal{F}',
            gG: '\\mathcal{G}',
            hH: '\\mathcal{H}',
        },
    },
});
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [
            ['$', '$'],
            ['\\(', '\\)'],
        ],
        processEscapes: true,
    },
});

/* Collapsed code block */

const collapsableCodeBlocks = document.querySelectorAll("div[class^='collapse'] .highlight");
for (var i = 0; i < collapsableCodeBlocks.length; i++) {
    const toggleContainer = document.createElement('div');
    toggleContainer.innerHTML = '<a href="#" class="toggle toggle-less" style="display:none;"><span class="icon icon-angle-double-up"></span><em>Show less...</em></a><a href="#" class="toggle toggle-more"><span class="icon icon-angle-double-down"></span><em>Show more...</em></a>';
    collapsableCodeBlocks[i].parentNode.insertBefore(toggleContainer, collapsableCodeBlocks[i].nextSibling);
}

const collapsableCodeToggles = document.querySelectorAll("div[class^='collapse'] .toggle");
for (var i = 0; i < collapsableCodeToggles.length; i++) {
    collapsableCodeToggles[i].addEventListener('click', function (e) {
        e.preventDefault();
        var codeBlock = this.closest('div[class^="collapse"]');
        if (codeBlock.classList.contains('expanded')) {
            codeBlock.classList.remove('expanded');
            this.style.display = 'none';
            this.nextSibling.style.display = 'block';
        } else {
            codeBlock.classList.add('expanded');
            this.style.display = 'none';
            this.previousSibling.style.display = 'block';
        }
    });
}

/* Wrap container around all tables allowing hirizontal scroll */

const contentTables = document.querySelectorAll('.content table');
for (var i = 0; i < contentTables.length; i++) {
    var wrapper = document.createElement('div');
    wrapper.classList.add('table-container');
    contentTables[i].parentNode.insertBefore(wrapper, contentTables[i]);
    wrapper.appendChild(contentTables[i]);
}
