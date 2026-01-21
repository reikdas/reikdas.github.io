const eleventySass = require("@grimlink/eleventy-plugin-sass");
const sass = require("sass");
const markdownIt = require("markdown-it");
const markdownItFootnote = require("markdown-it-footnote");

module.exports = function(eleventyConfig) {
    // set up Sass for compiling from *.scss to *.css
    eleventyConfig.addPlugin(eleventySass, { sass });

    // Configure markdown-it with footnote support
    const md = markdownIt({
        html: true,
        linkify: true,
        typographer: true
    }).use(markdownItFootnote);
    
    eleventyConfig.setLibrary("md", md);

    // Add a date filter for formatting dates
    eleventyConfig.addFilter("date", function(date, format) {
        if (!date) return "";
        const d = new Date(date);
        const months = ["January", "February", "March", "April", "May", "June",
                        "July", "August", "September", "October", "November", "December"];
        if (format === "%B %d, %Y") {
            return `${months[d.getMonth()]} ${d.getDate()}, ${d.getFullYear()}`;
        }
        return d.toLocaleDateString();
    });

    // Add a reverse filter for reversing arrays/collections
    eleventyConfig.addFilter("reverse", function(array) {
        if (!array) return [];
        return array.slice().reverse();
    });

    // Include any files in assets/ directly in your website without modifying
    // them. You can also use this for paper PDFs.
    //
    // See https://www.11ty.dev/docs/copy/
    eleventyConfig.addPassthroughCopy("assets/**");

    // You won't need this by default but if you're using a custom domain with
    // GitHub Pages, following the instructions at
    // https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/about-custom-domains-and-github-pages,
    // you're supposed to add a CNAME file with your custom domain. You can add
    // that to the root of your repository and this will set it up correctly.
    eleventyConfig.addPassthroughCopy("CNAME");

    return {
        // Your website might be hosted not at a root domain like
        // tchajed.github.io but a path like www.mit.edu/~tchajed (or
        // tchajed.github.com/personal-website-demo, like this setup is). This
        // pathPrefix is used by 11ty's url filter - see
        // https://www.11ty.dev/docs/filters/url/ for where this is useful. It's
        // included here mainly so you know how to change it if you need to.
        pathPrefix: "/",
        dir: {
            includes: "_templates",
            output: "_site",
        },
    };
};