# Folders

![Folder](/tutorial/folder.png)

Each folder inside the public repository correspond to a group of documentations. You can organize your files the way you want, and you have full control of what content you want to be visible or not.

The file `scribo.json` located in the root of the repository controls what documents you would like to show on your website. The property `docs` is an array of items with the following items:

| Property    | Type    | Description                                     |
| ----------- | ------- | ----------------------------------------------- |
| name        | string  | Documentation group name                        |
| description | string  | Description of the group                        |
| href        | string  | name of the folder                              |
| private     | boolean | if the group is private or not                  |
| repo        | string  | GitHub repository link, in case of private docs |

Example of a configuration:

```json
{
  ...
  "docs": [
    {
      "name": "Tutorial",
      "description": "Gettting Started with Scribo Platform",
      "href": "tutorial"
    },
    {
      "name": "Scribo API",
      "description": "Scribo REST API",
      "href": "scribo-api"
    },
    {
      "name": "Scribo Private Doc",
      "description": "Scribo Private Doc",
      "private": true,
      "href": "scribo-docs",
      "repo": "https://github.com/relferreira/scribo-docs"
    }
  ]
}

```
