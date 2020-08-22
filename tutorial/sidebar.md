# Sidebar

![Sidebar Construtor](/tutorial/sidebar.png)

Every Documentation has a `scribo.json` file where you can configure many options. One of them is how the sidebar will be presented. This json has a property called `sidebar`, that is an array of items with the following properties:

| Property | Type   | Description                       |
| -------- | ------ | --------------------------------- |
| type     | string | only supports the value: category |
| label    | string | Title of the section              |
| items    | array  | Menu items                        |

Each item has a label and an url

| Property | Type   | Description         |
| -------- | ------ | ------------------- |
| label    | string | Title of the menu   |
| href     | string | URL to the document |

Example of `scribo.json`:

```json
{
  "specificationUrl": "scribo-api/openapi.json",
  "sidebar": [
    {
      "type": "category",
      "label": "Documentation",
      "items": [
        {
          "label": "Commands",
          "href": "api-commands"
        }
      ]
    }
  ]
}
```
