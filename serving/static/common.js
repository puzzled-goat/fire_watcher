// Theme
const root = document.documentElement;
function initTheme() {
    if (localStorage.getItem("theme") === "dark") root.classList.add("dark");
}
function toggleTheme() {
    root.classList.toggle("dark");
    localStorage.setItem("theme", root.classList.contains("dark") ? "dark" : "light");
}

//  HEADER LOADER 
async function loadHeader() {
    const res = await fetch("/static/header.html");
    const html = await res.text();
    document.getElementById("header-container").innerHTML = html;

    // Theme toggle
    const themeToggleBtn = document.getElementById("theme-toggle");
    themeToggleBtn.addEventListener("click", toggleTheme);

    // Hamburger toggle
    const menuToggle = document.getElementById("menu-toggle");
    const navLinks = document.getElementById("nav-links");
    menuToggle.addEventListener("click", () => navLinks.classList.toggle("hidden"));

    //  Highlight active page 
    const page = document.body.dataset.page; // e.g., "home"
    const links = document.querySelectorAll("#nav-links a");
    links.forEach(link => {
        const href = link.getAttribute("href");
        if (!href.startsWith("/")) return; // skip external links
        const cleanHref = href.replace(/\/$/, "");
        if (cleanHref === "/" + page) {
            link.classList.add("bg-indigo-600", "text-white", "dark:bg-indigo-500");
        }
    });

    return {
        headerToggle: document.getElementById("header-toggle"),
        themeToggleBtn,
    };
}



//  FETCH JSON helper 
async function fetchJSON(url, options = {}) {
    const res = await fetch(url, options);
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
}

//  Scheduler toggle sync 
async function syncSchedulerToggles(headerToggle, bodyToggle, form) {
    const schedulerActive = headerToggle ? headerToggle.checked : bodyToggle.checked;

    if (headerToggle) headerToggle.checked = schedulerActive;
    if (bodyToggle) bodyToggle.checked = schedulerActive;

    if (form) {
        const formData = new FormData(form);
        formData.append("toggle_active", schedulerActive ? "true" : "false");
        await fetch("/scheduler/config", { method: "POST", body: formData });
    }
}


