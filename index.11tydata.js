// The module.exports object at the bottom is the data that powers the index.njk
// template file. When you start seeing {% %} and {{ }}, these are referring to
// objects in this file.
//
// This file is a JavaScript file that runs when the site is generated, which
// lets us flexibly prepare the data and simplifies the template.

// These are my frequent collaborators, so let's use some variables:
const me = "Pratyush Das";

// authorList generates the HTML for the author list from a JS array
function authorList(authors) {
    var list = [];
    authors.forEach((name, i) => {
        if (name == me) {
            name = '<span class="self-author">' + name + "</span>";
        }
        if (i == authors.length - 1) {
            list.push("and " + name);
        } else {
            list.push(name);
        }
    });
    return list.join(", ");
}

module.exports = {
    publications: [{
            title: "AwkwardForth: accelerating Uproot with an internal DSL",
            link: "https://www.epj-conferences.org/articles/epjconf/abs/2021/05/epjconf_chep2021_03002/epjconf_chep2021_03002.html",
            authors: authorList([
                "Jim Pivarski",
                "Ianna Osborne",
                me,
                "David Lange",
                "Peter Elmer",
            ]),
            conference: "25th International Conference on Computing in High-Energy and Nuclear Physics (vCHEP, 2021)",
        },
        {
            title: "Awkward Array: JSON-like data, NumPy-like idioms",
            link: "http://conference.scipy.org/proceedings/scipy2020/jim_pivarski.html",
            authors: authorList([
                "Jim Pivarski",
                "Ianna Osborne",
                me,
                "Anish Biswas",
                "Peter Elmer",
            ]),
            conference: "19th Python in Science Conference (SciPy USA, 2020)",
        },
        {
            title: "The Scikit HEP Project overview and prospects",
            link: "https://www.epj-conferences.org/articles/epjconf/abs/2020/21/epjconf_chep2020_06028/epjconf_chep2020_06028.html",
            authors: authorList([
                "Eduardo Rodrigues",
                "Benjamin Krikler",
                "Chris Burr",
                "Dmitri Smirnov",
                "Hans Dembinski",
                "Henry Schreiner",
                "Jaydeep Nandi",
                "Jim Pivarski",
                "Matthew Feickert",
                "Matthieu Marinangeli",
                "Nick Smith",
                me,
            ]),
            conference: "24th International Conference on Computing in High-Energy and Nuclear Physics (CHEP 2019)",
        },
    ],
    drafts: [{
            title: "DiSh: Dynamic Shell-Script Distribution",
            authors: authorList([
                "Tammam Mustafa",
                "Konstantinos Kallas",
                me,
                "Nikos Vasilakis",
            ]),
        },
    ],
    talks: [{
            title: "GSoC Experience - Enzyme",
            link: "https://youtu.be/mxI9fYbpndI",
            location: "LLVM Developers' Meeting, 2021",
        },
        {
            title: "Python in High Energy Physics",
            link: "https://youtu.be/jClVsR6XfdI",
            location: "PyCon USA, 2020",
        },
        {
            title: "Language Transformations for the Awkward Array library",
            link: "https://www.youtube.com/watch?v=yjlzO5oXb1w&list=PL-VTGB5hKxueocuuMS8ky9roL_QDqFX7m&index=2",
            location: "IRIS-HEP Fellow Presentations, 2020",
        },
        {
            title: "CUDA backend for the Awkward Array project",
            link: "https://drive.google.com/file/d/1AdNgteVH2gaUE3SKybeIXGUyqWlXoJJ9/view?usp=sharing",
            location: "Princeton University Liberty Research Group, 2020",
        },
        {
            title: "Python in High Energy Physics",
            link: "https://static.fossee.in/scipy2019/SciPyTalks/SciPyIndia2019_S008_Python_in_High_Energy_Physics_20191130.mp4",
            location: "SciPy India, 2019",
        },
        {
            title: "Writing files with uproot",
            link: "https://indico.cern.ch/event/833895/contributions/3577892/attachments/1927752/3191883/uproot-pyhep.pdf",
            location: "PyHEP, 2019",
        },
        {
            title: "Writing TTrees with uproot",
            link: "https://indico.cern.ch/event/840667/contributions/3527109/attachments/1908764/3153297/uproot-irisfellow-final.pdf",
            location: "IRIS-HEP: Summer student project presentations, 2019",
        },
        {
            title: "Writing files with uproot",
            link: "https://indico.cern.ch/event/697389/contributions/3102807/attachments/1713054/2762448/Writing_files_with_uproot.pdf",
            location: "ROOT Users' Workshop, 2018",
        },
        {
            title: "Writing files with uproot",
            link: "https://indico.cern.ch/event/754335/contributions/3166239/attachments/1734208/2804184/Writing_files_with_uproot_-_DIANA_HEP.pdf",
            location: "DIANA-HEP: Updates on ROOT I/O, 2018",
        },
        {
            title: "Separation of Concerns - ROOT4J and Spark-Root",
            link: "https://indico.cern.ch/event/658754/contributions/2685907/attachments/1506368/2347492/Refactoring_code_from_spark-root_to_root4j.pdf",
            location: "CMS Big Data Science Meeting, 2017",
        },
    ],
};
