# Theming

![Changing Doc Themes](/tutorial/themming.png)

Every Scribo compactible repository should have a `scribo.json` file. It is responsible for defining many configurations, like:
- Name (name)
- Logo Image (logoImg)
- Banner Title (bannerTitle)
- Banner Description (bannerDescription)
- Theme (theme)
- Menus (menus)

## Logo Image

Like the name implies, represents the image that will be positioned in the header of all pages. This property (logoImg) should be filled with the path to a `.png` file inside the `assets` folder, or a valid URL.

## Banner

Banner is a section of the Header only visible in the Home Page of the documentation webpage. This section contains a title and a description. 

We recommend you to make a breaf description of the documentations presented in the page

## Theme

Theme is an object with the following properties:

- Primary Color (primaryColor)
- Header Font Color (headerFontColor)

Primary Color is the main color used in the header, as a background, and as a tint color. Should represent your brand.

## Menus

List of Header links. We recommend some reference to your company main webpage and, maybe, a link to your carreer's contact.
Every Menu item should have:

- Label (label)
- Link (href)

## Example

```json

{
  "name": "Scribo",
  "logoImg": "logo.png",
  "bannerTitle": "Developer Hub",
  "bannerDescription": "Welcome to Scribo developer hub.",
  "theme": {
    "primaryColor": "#111a2b",
    "headerFontColor": "#fff"
  },
  "menus": [
    {
      "label": "About",
      "href": "https://scribo.dev"
    },
    {
      "label": "Carrer",
      "href": "https://scribo.dev"
    }
  ],
  ...
}

```

