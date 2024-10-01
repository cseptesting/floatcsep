/**
 * custom.js
 *
 * This script contains custom JavaScript modifications for the Sphinx documentation.
 * It can be expanded to include additional customizations related to behavior,
 * style, and functionality of the generated documentation.
 *
 *
 * Usage:
 * - Place this script in the _static directory of your Sphinx project.
 * - Include it in the html_js_files configuration in conf.py to load it automatically.
 * - Expand this file with other JavaScript customizations as needed.
 *
 * Author: Pablo Iturrieta
 * Date: 28.09.2024
 */

document.addEventListener("DOMContentLoaded", function () {
//     - Ensures that all external links open in a new tab by adding the target="_blank"
//       attribute to all links with the 'external' class (automatically applied by Sphinx).
//     - Adds rel="noopener noreferrer" for security purposes, ensuring the new page
//       does not have access to the originating window context (prevents security risks).
    // Select all external links in the documentation
    const links = document.querySelectorAll('a.external');

    // Loop through all the links and set them to open in a new tab
    links.forEach(function (link) {
        link.setAttribute('target', '_blank');
        link.setAttribute('rel', 'noopener noreferrer');
    });
});
